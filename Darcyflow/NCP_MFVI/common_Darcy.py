#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:45:50 2019

@author: jjx323
"""
import numpy as np
from scipy.special import gamma
import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
import dolfin as dl 
# import cupy as cp
# import cupyx.scipy.sparse as cpss
# import cupyx.scipy.sparse.linalg as cpssl

import sys, os
sys.path.append(os.pardir)
from core.probability import GaussianElliptic2
from core.model import ModelBase
from core.misc import my_project, trans2spnumpy, \
                      construct_measurement_matrix, make_symmetrize                     
from core.misc import load_expre, smoothing
from core.approximate_sample import LaplaceApproximate
from core.optimizer import NewtonCG
from core.linear_eq_solver import cg_my

                      

    
###########################################################################
class EquSolver(object):
    def __init__(self, domain_equ, f, m, points):
        self.domain_equ = domain_equ
        self.V_equ = self.domain_equ.function_space
        self.mm = fe.interpolate(m, self.V_equ)
        self.exp_m = fe.Function(self.V_equ)
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        self.f = fe.interpolate(f, self.V_equ)
        self.points = points
        
        self.u_, self.v_ = fe.TrialFunction(self.V_equ), fe.TestFunction(self.V_equ)
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.F_ = fe.assemble(self.f*self.v_*fe.dx)
        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V_equ, fe.Constant('0.0'), boundary)
        self.bc.apply(self.K_)
        self.bc.apply(self.F_)
        
        temp1 = fe.assemble(fe.inner(fe.Constant("1.0"), self.v_)*fe.dx)
        temp2 = temp1[:].copy()
        self.bc.apply(temp1)
        self.bc_idx = (temp2 != temp1)
        
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        self.F = self.F_[:]
        
        self.S = np.array(construct_measurement_matrix(points, self.V_equ).todense())
        
        ## All of the program did not highly rely on FEniCS, 
        ## so the following FEniCS function will be treated only as helpping function
        self.sol_forward = fe.Function(self.V_equ)
        self.sol_adjoint = fe.Function(self.V_equ)
        self.sol_incremental = fe.Function(self.V_equ)
        self.sol_incremental_adjoint = fe.Function(self.V_equ)
        self.Fs = fe.Function(self.V_equ)
        self.m_hat = fe.Function(self.V_equ)
        
        ## All of the solutions will be treated as the solution interact with 
        ## other program
        self.sol_forward_vec = self.sol_forward.vector()[:]
        self.sol_adjoint_vec = self.sol_adjoint.vector()[:]
        self.sol_incremental_vec = self.sol_incremental.vector()[:]
        self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint.vector()[:]
        
        self.is_cuda = False
        self.init_forward_sol, self.init_adjoint_sol = None, None
    
    def update_m(self, m_vec=None):
        if m_vec is None:
            self.mm.vector()[:] = 0.0
        else:
            self.mm.vector()[:] = np.array(m_vec)
        self.exp_m = fe.Function(self.V_equ)
        # self.exp_m.vector()[:] = fe.project(dl.exp(self.mm), self.V_equ).vector()[:]
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.bc.apply(self.K_)
        self.K = trans2spnumpy(self.K_)
        
        
        self.ff = 0
        self.rhs = 0
        
    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(self.points, self.domain_equ.function_space)
        self.S = np.array(self.S.todense())

        
    def get_data(self):
        if type(self.F) == np.ndarray:
            val = self.S@self.sol_forward.vector()[:]
            return np.array(val) 

    
    def to(self, device="cpu"):
        if device == "cpu":
            self.K, self.F = self.K.get(), self.F.get()
            self.S = self.S.get()
            self.sol_forward_vec = self.sol_forward_vec.get()
            self.sol_adjoint_vec = self.sol_adjoint_vec.get()
            self.sol_incremental_vec = self.sol_incremental_vec.get()
            self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint_vec.get()
            self.is_cuda = False

        else:
            raise NotImplementedError("device must be cpu or cuda")

    def forward_solver(self, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                fe.solve(self.K_, self.sol_forward.vector(), self.F_)
                self.sol_forward_vec = np.array(self.sol_forward.vector()[:])
            elif method == 'numpy':
                self.sol_forward_vec = spsl.spsolve(self.K, self.F)
                # self.sol_forward_vec = spsl.gmres(self.K, self.F, tol=1e-3)[0]
                self.sol_forward_vec = np.array(self.sol_forward_vec)

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_forward_vec
        
    def adjoint_solver(self, vec, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
            
        Fs = -self.S.T@vec
        Fs[self.bc_idx] = 0.0
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_adjoint.vector(), self.Fs.vector())
                self.sol_adjoint_vec = np.array(self.sol_adjoint.vector()[:])
            elif method == 'numpy':
                self.sol_adjoint_vec = np.array(spsl.spsolve(self.K, Fs))

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_adjoint_vec
  
    def incremental_forward_solver(self, m_hat, sol_forward=None, method='numpy'):
        if type(sol_forward) == type(None):
            self.sol_forward.vector()[:] = self.sol_forward_vec 
        
        if type(m_hat) == np.ndarray:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                b_ = -fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_forward), fe.grad(self.v_))*fe.dx)
                self.bc.apply(b_)
                fe.solve(self.K_, self.sol_incremental.vector(), b_)
                self.sol_incremental_vec = np.array(self.sol_incremental.vector()[:])
            elif method == 'numpy':
                b_ = fe.inner(self.exp_m*fe.grad(self.sol_forward)*self.u_, fe.grad(self.v_))*fe.dx
                b_ = fe.assemble(b_)   
                b_spnumpy = trans2spnumpy(b_)
                b = b_spnumpy@m_hat
                self.sol_incremental_vec = np.array(spsl.spsolve(self.K, -b))

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_incremental_vec     
        
    def incremental_adjoint_solver(self, vec, m_hat, sol_adjoint=None, simple=False, method='numpy'):
        if type(sol_adjoint) == type(None):
            self.sol_adjoint.vector()[:] = self.sol_adjoint_vec 
        
        Fs = -self.S.T@vec
        Fs = Fs.squeeze()
        if simple == False:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                bl_ = fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_adjoint), fe.grad(self.v_))*fe.dx)
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), -bl_ + self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                bl_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.sol_adjoint)*self.u_, fe.grad(self.v_))*fe.dx)
                bl_spnumpy = trans2spnumpy(bl_)
                if type(m_hat) == np.ndarray:
                    bl = bl_spnumpy@m_hat
                    # print(bl.shape, Fs.shape)
                    rhs = -bl + Fs
                    rhs[self.bc_idx] = 0.0
                    self.sol_incremental_adjoint_vec = spsl.spsolve(self.K, rhs)
                else:
                    raise NotImplementedError("device must be cpu or cuda")
        elif simple == True:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                if type(vec) == np.ndarray:
                    Fs[self.bc_idx] = 0.0
                    val = spsl.spsolve(self.K, Fs)
                    self.sol_incremental_adjoint_vec = np.array(val)
                
                else:
                    raise NotImplementedError("device must be cpu or cuda")
                
        return self.sol_incremental_adjoint_vec

########################################################### Linear part

class GaussianLam(object):
    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov
        
    def generate_sample(self):
        return np.random.normal(self.mean, np.sqrt(self.cov))

class EquSolver_linear(object):
    def __init__(self, domain_equ, mMap, points, p0):
        '''
        
        This sovler is used to solve the linear equation.
        The parameter mMap = m_Map; p0 = H(m_Map)
        
        '''
        self.domain_equ = domain_equ
        self.V_equ = self.domain_equ.function_space
        self.m = fe.Function(self.V_equ)
        
        self.mMap_fun = fe.Function(self.V_equ)
        self.mMap_fun.vector()[:] = mMap
        self.mm = fe.interpolate(self.mMap_fun, self.V_equ)
        self.exp_m = fe.Function(self.V_equ)
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        # self.f_fun = fe.Function(self.V_equ)
        # self.f_fun.vector()[:] = f
        # self.f = self.f_fun.vector()[:]
        self.p0_fun = fe.Function(self.V_equ)
        self.p0_fun.vector()[:] = p0
        self.p0 = self.p0_fun.vector()[:]
        self.points = points
        
        self.u_, self.v_ = fe.TrialFunction(self.V_equ), fe.TestFunction(self.V_equ)
        self.K_ = fe.assemble(-fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        # self.F_ = fe.assemble(self.f*self.v_*fe.dx)
        self.F_ = fe.assemble(fe.inner(self.exp_m * fe.grad(self.p0_fun) * self.u_, fe.grad(self.v_)) * fe.dx)

        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V_equ, fe.Constant('0.0'), boundary)
        self.bc.apply(self.K_)
        self.bc.apply(self.F_)
        
        temp1 = fe.assemble(fe.inner(fe.Constant("1.0"), self.v_)*fe.dx)
        temp2 = temp1[:].copy()
        self.bc.apply(temp1)
        self.bc_idx = (temp2 != temp1)
        
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        # self.F = self.F_[:]
        self.F = trans2spnumpy(self.F_)
        self.F[self.bc_idx, :] = 0.0
        self.S = np.array(construct_measurement_matrix(points, self.V_equ).todense())
        
        ## All of the program did not highly rely on FEniCS, 
        ## so the following FEniCS function will be treated only as helpping function
        self.sol_forward = fe.Function(self.V_equ)
        self.sol_adjoint = fe.Function(self.V_equ)
        self.sol_incremental = fe.Function(self.V_equ)
        self.sol_incremental_adjoint = fe.Function(self.V_equ)
        self.Fs = fe.Function(self.V_equ)
        self.m_hat = fe.Function(self.V_equ)
        
        ## All of the solutions will be treated as the solution interact with 
        ## other program
        self.sol_forward_vec = self.sol_forward.vector()[:]
        self.sol_adjoint_vec = self.sol_adjoint.vector()[:]
        self.sol_incremental_vec = self.sol_incremental.vector()[:]
        self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint.vector()[:]
        
        self.is_cuda = False
        self.init_forward_sol, self.init_adjoint_sol = None, None
    
    def update_m(self, m_vec):
        assert len(self.m.vector()[:]) == len(m_vec) 
        self.m.vector()[:] = np.array(m_vec[:])
        
     
    def forward_solver(self, m_vec=None):
        if m_vec is not None:
            self.update_m(m_vec)
        
        rhs = self.F @ self.m.vector()[:]
        
        self.sol_forward_vec = spsl.spsolve(self.K, rhs)
        self.sol_forward_vec = np.array(self.sol_forward_vec)
        
        return self.sol_forward_vec
    
    def incremental_forward_solver(self, m_hat=None):
        ## we need this function can accept matrix input
        ## For linear problems, the incremental forward == forward
        rhs = self.F @ m_hat

        self.sol_incremental_vec = spsl.spsolve(self.K, rhs)
        self.sol_incremental_vec = np.array(self.sol_incremental_vec)
        
        return self.sol_incremental_vec
    
    def adjoint_solver(self, res_vec):
        
        rhs = self.S.T @ res_vec
        
        
        self.sol_adjoint_vec = spsl.spsolve(self.M, self.F.T@spsl.spsolve(self.K.T, rhs))
        self.sol_adjoint_vec = np.array(self.sol_adjoint_vec) 
        
        return self.sol_adjoint_vec
    
    
    def incremental_adjoint_solver(self, vec, m_hat=None):  
        
        self.sol_incremental_adjoint_vec = self.adjoint_solver(vec)
        self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint_vec)
        
        return self.sol_incremental_adjoint_vec
        
########################################################### equ_solver hybird
class EquSolver_hybird(object):
    def __init__(self, domain_equ, f, m, points, mMap, p0):
        self.domain_equ = domain_equ
        self.V_equ = self.domain_equ.function_space
        self.m = m
        self.mm = fe.interpolate(m, self.V_equ)
        self.exp_m = fe.Function(self.V_equ)
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        self.f = fe.interpolate(f, self.V_equ)
        self.points = points
        
        self.u_, self.v_ = fe.TrialFunction(self.V_equ), fe.TestFunction(self.V_equ)
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.F_ = fe.assemble(self.f*self.v_*fe.dx)
        self.M_ = fe.assemble(fe.inner(self.u_, self.v_)*fe.dx)
        
        self.mMap_fun = fe.Function(self.V_equ)
        self.mMap_fun.vector()[:] = mMap
        self.mm = fe.interpolate(self.mMap_fun, self.V_equ)
        self.exp_mMap = fe.Function(self.V_equ)
        self.exp_mMap.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        # self.f_fun = fe.Function(self.V_equ)
        # self.f_fun.vector()[:] = f
        # self.f = self.f_fun.vector()[:]
        self.p0_fun = fe.Function(self.V_equ)
        self.p0_fun.vector()[:] = p0
        self.p0 = self.p0_fun.vector()[:]
        
        # self.KK_ = fe.assemble(-fe.inner(self.exp_mMap*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)   
        # self.FF_ = fe.assemble(fe.inner(self.exp_mMap * fe.grad(self.p0_fun) * self.u_, fe.grad(self.v_)) * fe.dx)
        self.KK_ = fe.assemble(-fe.inner(fe.grad(self.u_), fe.grad(self.v_))*fe.dx)   
        self.FF_ = fe.assemble(fe.inner(fe.grad(self.p0_fun) * self.u_, fe.grad(self.v_)) * fe.dx)

        
        
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = fe.DirichletBC(self.V_equ, fe.Constant('0.0'), boundary)
        self.bc.apply(self.K_)
        self.bc.apply(self.F_)
        self.bc.apply(self.KK_)
        self.bc.apply(self.FF_)
        
        temp1 = fe.assemble(fe.inner(fe.Constant("1.0"), self.v_)*fe.dx)
        temp2 = temp1[:].copy()
        self.bc.apply(temp1)
        self.bc_idx = (temp2 != temp1)
        
        self.K = trans2spnumpy(self.K_)
        self.M = trans2spnumpy(self.M_)
        self.F = self.F_[:]
        self.KK = trans2spnumpy(self.KK_)
        self.FF = trans2spnumpy(self.FF_)
        
        self.S = np.array(construct_measurement_matrix(points, self.V_equ).todense())
        
        ## All of the program did not highly rely on FEniCS, 
        ## so the following FEniCS function will be treated only as helpping function
        self.sol_forward = fe.Function(self.V_equ)
        self.sol_adjoint = fe.Function(self.V_equ)
        self.sol_incremental = fe.Function(self.V_equ)
        self.sol_incremental_adjoint = fe.Function(self.V_equ)
        self.Fs = fe.Function(self.V_equ)
        self.m_hat = fe.Function(self.V_equ)
        
        ## All of the solutions will be treated as the solution interact with 
        ## other program
        self.sol_forward_vec = self.sol_forward.vector()[:]
        self.sol_adjoint_vec = self.sol_adjoint.vector()[:]
        self.sol_incremental_vec = self.sol_incremental.vector()[:]
        self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint.vector()[:]
        
        self.is_cuda = False
        self.init_forward_sol, self.init_adjoint_sol = None, None
    
    def update_m(self, m_vec=None):
        if m_vec is None:
            self.mm.vector()[:] = 0.0
        else:
            self.mm.vector()[:] = np.array(m_vec)
        self.exp_m = fe.Function(self.V_equ)
        # self.exp_m.vector()[:] = fe.project(dl.exp(self.mm), self.V_equ).vector()[:]
        self.exp_m.vector()[:] = my_project(dl.exp(self.mm), self.V_equ, flag='only_vec')
        
        self.K_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.u_), fe.grad(self.v_))*fe.dx)
        self.bc.apply(self.K_)
        self.K = trans2spnumpy(self.K_)
        
        
        self.ff = 0
        self.rhs = 0
        
    def update_points(self, points):
        self.points = points
        self.S = construct_measurement_matrix(self.points, self.domain_equ.function_space)
        self.S = np.array(self.S.todense())

        
    def get_data(self):
        if type(self.F) == np.ndarray:
            val = self.S@self.sol_forward.vector()[:]
            return np.array(val) 

    
    def to(self, device="cpu"):
        if device == "cpu":
            self.K, self.F = self.K.get(), self.F.get()
            self.S = self.S.get()
            self.sol_forward_vec = self.sol_forward_vec.get()
            self.sol_adjoint_vec = self.sol_adjoint_vec.get()
            self.sol_incremental_vec = self.sol_incremental_vec.get()
            self.sol_incremental_adjoint_vec = self.sol_incremental_adjoint_vec.get()
            self.is_cuda = False

        else:
            raise NotImplementedError("device must be cpu or cuda")

    def forward_solver(self, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                fe.solve(self.K_, self.sol_forward.vector(), self.F_)
                self.sol_forward_vec = np.array(self.sol_forward.vector()[:])
            elif method == 'numpy':
                self.sol_forward_vec = spsl.spsolve(self.K, self.F)
                # self.sol_forward_vec = spsl.gmres(self.K, self.F, tol=1e-3)[0]
                self.sol_forward_vec = np.array(self.sol_forward_vec)

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_forward_vec
        
    def adjoint_solver(self, vec, m_vec=None, method='numpy'):
        if type(m_vec) != type(None):
            self.update_m(m_vec)
            
        Fs = -self.S.T@vec
        Fs[self.bc_idx] = 0.0
        
        if type(self.F) == np.ndarray:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_adjoint.vector(), self.Fs.vector())
                self.sol_adjoint_vec = np.array(self.sol_adjoint.vector()[:])
            elif method == 'numpy':
                self.sol_adjoint_vec = np.array(spsl.spsolve(self.K, Fs))

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_adjoint_vec
  
    def incremental_forward_solver(self, m_hat, sol_forward=None, method='numpy'):
        if type(sol_forward) == type(None):
            self.sol_forward.vector()[:] = self.sol_forward_vec 
        
        if type(m_hat) == np.ndarray:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                b_ = -fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_forward), fe.grad(self.v_))*fe.dx)
                self.bc.apply(b_)
                fe.solve(self.K_, self.sol_incremental.vector(), b_)
                self.sol_incremental_vec = np.array(self.sol_incremental.vector()[:])
            elif method == 'numpy':
                b_ = fe.inner(self.exp_m*fe.grad(self.sol_forward)*self.u_, fe.grad(self.v_))*fe.dx
                b_ = fe.assemble(b_)   
                b_spnumpy = trans2spnumpy(b_)
                b = b_spnumpy@m_hat
                self.sol_incremental_vec = np.array(spsl.spsolve(self.K, -b))

        else:
            raise NotImplementedError("device must be cpu or cuda")
            
        return self.sol_incremental_vec     
        
    def incremental_adjoint_solver(self, vec, m_hat, sol_adjoint=None, simple=False, method='numpy'):
        if type(sol_adjoint) == type(None):
            self.sol_adjoint.vector()[:] = self.sol_adjoint_vec 
        
        Fs = -self.S.T@vec
        Fs = Fs.squeeze()
        if simple == False:
            if method == 'FEniCS':
                self.m_hat.vector()[:] = np.array(m_hat)
                bl_ = fe.assemble(fe.inner(self.m_hat*self.exp_m*fe.grad(self.sol_adjoint), fe.grad(self.v_))*fe.dx)
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), -bl_ + self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                bl_ = fe.assemble(fe.inner(self.exp_m*fe.grad(self.sol_adjoint)*self.u_, fe.grad(self.v_))*fe.dx)
                bl_spnumpy = trans2spnumpy(bl_)
                if type(m_hat) == np.ndarray:
                    bl = bl_spnumpy@m_hat
                    # print(bl.shape, Fs.shape)
                    rhs = -bl + Fs
                    rhs[self.bc_idx] = 0.0
                    self.sol_incremental_adjoint_vec = spsl.spsolve(self.K, rhs)
                else:
                    raise NotImplementedError("device must be cpu or cuda")
        elif simple == True:
            if method == 'FEniCS':
                self.Fs.vector()[:] = Fs
                fe.solve(self.K_, self.sol_incremental_adjoint.vector(), self.Fs.vector())
                self.sol_incremental_adjoint_vec = np.array(self.sol_incremental_adjoint.vector()[:])
            elif method == 'numpy':
                if type(vec) == np.ndarray:
                    Fs[self.bc_idx] = 0.0
                    val = spsl.spsolve(self.K, Fs)
                    self.sol_incremental_adjoint_vec = np.array(val)
                
                else:
                    raise NotImplementedError("device must be cpu or cuda")
                
        return self.sol_incremental_adjoint_vec
    
    def forward_solver_linear(self, m_vec):

        
        rhs = self.FF @ m_vec
        rhs[self.bc_idx] = 0.0 
        
        self.sol_forward_vec = spsl.spsolve(self.KK, rhs)
        self.sol_forward_vec = np.array(self.sol_forward_vec)
        
        return self.sol_forward_vec
        
###########################################################################
class ModelDarcyFlow(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
        self.p = fe.Function(self.equ_solver.domain_equ.function_space)
        self.q = fe.Function(self.equ_solver.domain_equ.function_space)
        self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
        self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
        self.u_ = fe.TrialFunction(self.domain_equ.function_space)
        self.v_ = fe.TestFunction(self.domain_equ.function_space)
        self.m_hat = fe.Function(self.domain_equ.function_space)
        self.m = self.equ_solver.mm
        self.loss_residual_now = 0

    def update_m(self, m_vec, update_sol=True):
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()

    
    def updata_d(self, d):
        self.d = d
        
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None):
            val = self.noise.precision@vec
        else:
            val = spsl.spsolve(self.noise.covariance, vec)
        return np.array(val)
        
    def loss_residual(self, m_vec=None):
        if m_vec is None:
            temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()
        else:
            self.update_m(m_vec, update_sol=True)
            temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()

        temp = (temp - self.noise.mean - self.d)
        if type(self.noise.precision) != type(None): 
            temp = temp@(self.noise.precision)@temp
        else:
            temp = temp@(spsl.spsolve(self.noise.covariance, temp))
        self.loss_residual_now = 0.5*temp
        return self.loss_residual_now
    
    def loss_residual_L2(self):
        temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
        temp = temp@temp
        return 0.5*temp
    
    def eval_grad_residual(self, m_vec):
        self.update_m(m_vec, update_sol=False)
        self.equ_solver.forward_solver()
        vec = np.array(self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d)
        vec = self._time_noise_precision(vec.squeeze()) 
        self.equ_solver.adjoint_solver(vec)
        self.p.vector()[:] = self.equ_solver.sol_forward_vec
        self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
        b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
        return spsl.spsolve(self.equ_solver.M, b_[:])

        
    def eval_hessian_res_vec(self, m_hat_vec):
        # self.m_hat.vector()[:] = m_hat_vec
        self.equ_solver.incremental_forward_solver(m_hat_vec)
        vec = np.array(self.S@self.equ_solver.sol_incremental_vec)
        vec = self._time_noise_precision(vec.squeeze())        
        self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
        self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
        self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
        A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
                                  fe.grad(self.q))*fe.dx)
        A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
                                  fe.grad(self.qq))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
        A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
                                  fe.grad(self.pp))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
        A = A1[:] + A2[:] + A3[:]
        
        return spsl.spsolve(self.equ_solver.M, A)

########################################################### Linear part

class ModelDarcyFlow_linear(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver, lam_dis):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        self.u_ = fe.TrialFunction(self.domain_equ.function_space)
        self.v_ = fe.TestFunction(self.domain_equ.function_space)
    
    def update_m(self, m_vec, update_sol=True):
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
            
    
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.mean
            
    def loss_residual(self, m_vec=None):
        
        if m_vec is not None:
            self.update_m(m_vec, update_sol=False)
        ## scale lam
        cc = self.lam*self.lam + self.lam_dis.cov
        
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        temp = (self.S@self.p.vector()[:])
        temp = (temp - self.noise.mean - self.d)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        return 0.5*cc*temp

    def loss_residual_L2(self):
        ## scale lam
        cc = self.lam*self.lam + self.lam_dis.cov
        
        temp = (self.S@self.p.vector()[:])
        temp = (temp - self.d)
        temp = temp@temp
        return 0.5*cc*temp
    
    def eval_HAdjointData(self):
        res_vec = spsl.spsolve(self.noise.covariance, self.d)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = self.lam*g_.vector()[:]
        return np.array(g)
    
    def eval_grad_residual(self, m_vec):
        cc = self.lam*self.lam + self.lam_dis.cov
        
        self.equ_solver.update_m(m_vec)
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        ## scale lam
        measure_scale = self.S@(self.p.vector()[:])
        res_vec = spsl.spsolve(self.noise.covariance, measure_scale - self.d)
        # print(res_vec)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        ## scale lam
        g = g_.vector()[:]
        return np.array(cc*g)

    
    def eval_hessian_res_vec(self, dm):
        cc = self.lam*self.lam + self.lam_dis.cov
        
        self.equ_solver.update_m(dm)
        self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        measure = (self.S@(self.p.vector()[:]))
        res_vec = spsl.spsolve(self.noise.covariance, measure)
        self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        HM = g_.vector()[:]
        return np.array(cc*HM)
    
    def update_lam_dis(self, m, eigval):
        tmp = self.lam_dis.cov + self.lam_dis.mean*self.lam_dis.mean
        eigval = eigval/tmp
        # update covariance
        # self.p.vector()[:] = self.equ_solver.forward_solver(m)
        mm = m - self.equ_solver.mMap_fun.vector()[:]
        mm = mm * self.lam_dis.mean / tmp
        self.p.vector()[:] = self.equ_solver.forward_solver(mm)
        p = self.p.vector()[:]
        Sp = self.S@p
        if self.noise.precision is None:
            temp = Sp@spsl.spsolve(self.noise.covariance, Sp)
        else:
            temp = Sp@self.noise.precision@Sp
        
        # print('^^^^^^^^^', np.max(p))
        
        temp1 = temp + 1/self.lam_cov0
        temp2 = np.sum(eigval/(tmp*eigval + 1)) / tmp
        self.lam_dis.cov = 1/(temp1 + temp2)
        print("-----", temp2)
        
        # update mean
        if self.noise.precision is None:
            tmp = self.d@spsl.spsolve(self.noise.covariance, Sp)
        else:
            tmp = self.d@self.noise.precision@(Sp)
        print("*****", tmp, temp)
            
        self.lam_dis.mean = self.lam_dis.cov*(tmp + self.lam_mean0/self.lam_cov0)
        # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
    
    
    def update_lam_dis_(self, m, eigval):
        
        
        rho = self.lam_dis.mean * self.lam_dis.mean + self.lam_dis.cov
        vk = m / np.sqrt(rho)
        Hvk = self.equ_solver.S @ np.array(self.equ_solver.forward_solver(vk)).squeeze()
        Hvk2 = Hvk @ spsl.spsolve(self.noise.covariance, Hvk)
        temp1 = Hvk2 + 1 / self.lam_cov0
        
        eigval = eigval / rho
        temp2 = np.sum(eigval/(rho*eigval + 1))
        # temp2 = 0
        self.lam_dis.cov = 1/(temp1 + temp2)
        
        tmp1 = self.d @spsl.spsolve(self.noise.covariance, Hvk)
        self.lam_dis.mean = self.lam_dis.cov * (tmp1 + self.lam_mean0/self.lam_cov0)
        
        # print('*********', tmp1, temp2)
        # print('lam_mean, lam_cov, = ', self.lam_dis.mean, self.lam_dis.cov)
        
################################################################## model hybird
class ModelDarcyFlow_hybird(ModelBase):
    def __init__(self, d, domain_equ, prior, noise, equ_solver, lam_dis, d_linear):
        super().__init__(d, domain_equ, prior, noise, equ_solver)
        self.p = fe.Function(self.equ_solver.domain_equ.function_space)
        self.q = fe.Function(self.equ_solver.domain_equ.function_space)
        self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
        self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
        self.u_ = fe.TrialFunction(self.domain_equ.function_space)
        self.v_ = fe.TestFunction(self.domain_equ.function_space)
        self.m_hat = fe.Function(self.domain_equ.function_space)
        self.m = self.equ_solver.mm
        self.loss_residual_now = 0
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        self.d_linear = d_linear

    def update_m(self, m_vec, update_sol=True):
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
            
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.mean
    
    def updata_d(self, d):
        self.d = d
        
    def _time_noise_precision(self, vec):
        if type(self.noise.precision) != type(None):
            val = self.noise.precision@vec
        else:
            val = spsl.spsolve(self.noise.covariance, vec)
        return np.array(val)
        
    def loss_residual(self, m_vec=None):
        if m_vec is None:
            temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()
        else:
            self.update_m(m_vec, update_sol=True)
            temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()

        temp = (temp - self.noise.mean - self.d)
        if type(self.noise.precision) != type(None): 
            temp = temp@(self.noise.precision)@temp
        else:
            temp = temp@(spsl.spsolve(self.noise.covariance, temp))
        self.loss_residual_now = 0.5*temp
        tmp = self.lam_dis.mean**2 + self.lam_dis.cov
        return tmp*self.loss_residual_now
    
    def loss_residual_L2(self):
        temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
        temp = temp@temp
        tmp = self.lam_dis.mean**2 + self.lam_dis.cov
        return 0.5*tmp*temp
    
    def eval_grad_residual(self, m_vec):
        self.update_m(m_vec, update_sol=False)
        self.equ_solver.forward_solver()
        vec = np.array(self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d)
        vec = self._time_noise_precision(vec.squeeze()) 
        self.equ_solver.adjoint_solver(vec)
        self.p.vector()[:] = self.equ_solver.sol_forward_vec
        self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
        b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
        tmp = self.lam_dis.mean ** 2 + self.lam_dis.cov
        return tmp*spsl.spsolve(self.equ_solver.M, b_[:])

        
    def eval_hessian_res_vec(self, m_hat_vec):
        # self.m_hat.vector()[:] = m_hat_vec
        self.equ_solver.incremental_forward_solver(m_hat_vec)
        vec = np.array(self.S@self.equ_solver.sol_incremental_vec)
        vec = self._time_noise_precision(vec.squeeze())        
        self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
        self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
        self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
        A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
                                  fe.grad(self.q))*fe.dx)
        A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
                                  fe.grad(self.qq))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
        A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
                                  fe.grad(self.pp))*fe.dx)
                         # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
        A = A1[:] + A2[:] + A3[:]
        
        tmp = self.lam_dis.mean**2 + self.lam_dis.cov
        
        return tmp*spsl.spsolve(self.equ_solver.M, A)
    
    
    def update_lam_dis(self, m, eigval):
        tmp = self.lam_dis.cov + self.lam_dis.mean*self.lam_dis.mean
        eigval = eigval/tmp
        # update covariance
        # self.p.vector()[:] = self.equ_solver.forward_solver(m)
        mm = m - self.equ_solver.mMap_fun.vector()[:]
        mm = mm / np.sqrt(tmp)
        self.p.vector()[:] = self.equ_solver.forward_solver_linear(mm)
        p = self.p.vector()[:]
        Sp = self.S@p
        if self.noise.precision is None:
            temp = Sp@spsl.spsolve(self.noise.covariance, Sp)
        else:
            temp = Sp@self.noise.precision@Sp
        
        temp1 = temp + 1/self.lam_cov0
        temp2 = np.sum(eigval/(tmp*eigval + 1))
        self.lam_dis.cov = 1/(temp1 + temp2)
        # print("-----", temp1, temp2)
        
        # update mean
        # if self.noise.precision is None:
        #     tmp = self.d@spsl.spsolve(self.noise.covariance, Sp)
        # else:
        #     tmp = self.d@self.noise.precision@(Sp)
        
        if self.noise.precision is None:
            tmp = self.d_linear @spsl.spsolve(self.noise.covariance, Sp)
        else:
            tmp = self.d_linear @self.noise.precision@(Sp)
        
        print("*****", tmp, temp)
            
        self.lam_dis.mean = self.lam_dis.cov*(tmp + self.lam_mean0/self.lam_cov0)
        
        

##################################################################
class PosteriorOfV(LaplaceApproximate):
    def __init__(self, model, newton_cg):
        super().__init__(model)
        self.model = model
        self.hessian_operator = self.model.MxHessian_linear_operator()
        self.newton_cg = newton_cg

    def calculate_eigensystem_lam(self, num_eigval, method='double_pass', 
                              oversampling_factor=20, cut_val=0.9, **kwargs):
        ## the calculate_eigensystem calculate the eigensystem of 
        ## the operator C_0^{1/2}H_{misfit}C_0^{1/2} 
        self.calculate_eigensystem(num_eigval, method, oversampling_factor, 
                                   cut_val, **kwargs)
        
        tmp = self.model.lam_dis.mean**2 + self.model.lam_dis.cov
        
        self.eigval_lam = self.eigval / tmp
     
    def pointwise_variance_field_lam(self, xx, yy):
        assert hasattr(self, "eigval") and hasattr(self, "eigvec")
        
        SN = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = construct_measurement_matrix(np.array(xx), self.prior.domain.function_space)
        SM = SM.T
        
        val = spsl.spsolve(self.prior.K, SM)
        dr = self.eigval_lam/(self.eigval_lam + 1.0)
        Dr = sps.csc_matrix(sps.diags(dr))
        val1 = self.eigvec@Dr@self.eigvec.T@self.M@val
        val = self.M@(val - val1)
        val = spsl.spsolve(self.prior.K, val)
        val = SN@val     
        
        if type(val) == type(self.M):
            val = val.todense()
        
        return np.array(val)  
    
    def generate_sample_lam(self):
        assert hasattr(self, "mean") and hasattr(self, "eigval") and hasattr(self, "eigvec")
        n = np.random.normal(0, 1, (self.fun_dim,))
        val1 = self.Minv_lamped_half@n
        pr = 1.0/np.sqrt(self.eigval_lam+1.0) - 1.0
        Pr = sps.csc_matrix(sps.diags(pr))
        val2 = self.eigvec@Pr@self.eigvec.T@self.M@val1
        val = self.M@(val1 + val2)
        val = spsl.spsolve(self.prior.K, val)
        val = self.mean + val
        return np.array(val)
     
    def eval_mean_iterative(self, iter_num=50, cg_max=1000, method="cg_my"):
        
        
        ## Without a good initial value, it seems hard for us to obtain a good solution
        # init_fun = smoothing(self.equ_solver.m, alpha=0.1)
        # newton_cg.re_init(init_fun.vector()[:])

        ## calculate the posterior mean 
        max_iter = iter_num
        # loss_pre = self.model.loss()[0]
        # for itr in range(max_iter):
        #     self.newton_cg.descent_direction(cg_max=cg_max, method=method)
        #     # newton_cg.descent_direction(cg_max=30, method='bicgstab')
        #     # print(newton_cg.hessian_terminate_info)
        #     self.newton_cg.step(method='armijo', show_step=False)
        #     if self.newton_cg.converged == False:
        #         break
        #     loss = self.model.loss()[0]
        #     # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        #     if np.abs(loss - loss_pre) < 1e-3*loss:
        #         # print("Iteration stoped at iter = %d" % itr)
        #         break 
        #     loss_pre = loss
            
        loss_pre = self.model.loss()[0]
        for itr in range(max_iter):
            self.newton_cg.descent_direction(cg_max=30, method='cg_my')
            # newton_cg.descent_direction(cg_max=30, method='bicgstab')
            # print(newton_cg.hessian_terminate_info)
            self.newton_cg.step(method='armijo', show_step=False)
            # if newton_cg.converged == False:
            #     break
            loss = self.model.loss()[0]
            print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-3*loss:
                print("Iteration stoped at iter = %d" % itr)
                break 
            loss_pre = loss
        
        
        rho = self.model.lam_dis.mean*self.model.lam_dis.mean + self.model.lam_dis.cov

        self.mean = np.array(self.newton_cg.mk.copy()) * self.model.lam_dis.mean / np.sqrt(rho)
        

        
    def eval_mean(self, cg_tol=None, cg_max=1000, method='cg_my', curvature_detector=False):
        self.g = self.model.eval_HAdjointData()
        pre_cond = self.model.precondition_linear_operator()
            
        if cg_tol is None:
            ## if cg_tol is None, specify cg_tol according to the rule in the following paper:
            ## Learning physics-based models from data: perspective from inverse problems
            ## and model reduction, Acta Numerica, 2021. Page: 465-466
            norm_grad = np.sqrt(self.g@self.M@self.g)
            cg_tol = min(0.5, np.sqrt(norm_grad))
        atol = 0.1
        ## cg iteration will terminate when norm(residual) <= min(atol, cg_tol*|self.grad|)
             
        if method == 'cg_my':
            ## Note that the M and Minv option of cg_my are designed different to the methods 
            ## in scipy, e.g., Minv should be M in scipy
            ## Here, in order to keep the matrix to be symmetric and positive definite,
            ## we actually need to solve M H g = M (-grad) instead of H g = -grad
            self.mean, info, k = cg_my(
                self.hessian_operator, self.M@self.g, Minv=pre_cond,
                tol=cg_tol, atol=atol, maxiter=cg_max, curvature_detector=True
                )
            # if k == 1:  
            #     ## If the curvature is negative for the first iterate, use grad as the 
            #     ## search direction.
            #     self.g = -self.grad
            # # print(info)
        elif method == 'bicgstab':
            ## It seems hardly to incoporate the curvature terminate condition 
            ## into the cg type methods in scipy. However, we keep the implementation.
            ## Since there may be some ways to add curvature terminate condition, and 
            ## cg type methods in scipy can be used as a basedline for testing linear problems.
            self.mean, info = spsl.bicgstab(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cg':
            self.mean, info = spsl.cg(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )
        elif method == 'cgs':
            self.mean, info = spsl.cgs(
                self.hessian_operator, self.M@self.g, M=pre_cond, tol=cg_tol, atol=atol, maxiter=cg_max,
                callback=None
                )       
        else:
            assert False, "method should be cg, cgs, bicgstab"
        
        self.hessian_terminate_info = info


# ###########################################################################
# class ModelNoPrior(ModelBase):
#     def __init__(self, d, domain_equ, noise, equ_solver, prior=None):
#         super().__init__(d, domain_equ, prior, noise, equ_solver)
#         self.p = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.q = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.pp = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.qq = fe.Function(self.equ_solver.domain_equ.function_space)
#         self.u_ = fe.TrialFunction(self.domain_equ.function_space)
#         self.v_ = fe.TestFunction(self.domain_equ.function_space)
#         self.m_hat = fe.Function(self.domain_equ.function_space)
#         self.m = self.equ_solver.mm
#         self.loss_residual_now = 0

#     def update_m(self, m_vec, update_sol=True):
#         self.equ_solver.update_m(m_vec)
#         if update_sol == True:
#             self.equ_solver.forward_solver()
    
#     def updata_d(self, d):
#         self.d = d
        
#     def _time_noise_precision(self, vec):
#         if type(self.noise.precision) != type(None):
#             val = self.noise.precision@vec
#         else:
#             val = spsl.spsolve(self.noise.covariance, vec)
#         return np.array(val)
        
#     def loss_residual(self):
#         temp = np.array(self.S@self.equ_solver.sol_forward_vec).flatten()
#         temp = (temp - self.noise.mean - self.d)
#         if type(self.noise.precision) != type(None): 
#             temp = temp@(self.noise.precision)@temp
#         else:
#             temp = temp@(spsl.spsolve(self.noise.covariance, temp))
#         self.loss_residual_now = 0.5*temp
#         return self.loss_residual_now
    
#     def loss_residual_L2(self):
#         temp = (self.S@self.equ_solver.sol_forward_vec - self.d)
#         temp = temp@temp
#         return 0.5*temp
    
#     def eval_grad_residual(self, m_vec):
#         self.update_m(m_vec, update_sol=False)
#         self.equ_solver.forward_solver()
#         vec = self.S@self.equ_solver.sol_forward_vec - self.noise.mean - self.d
#         vec = self._time_noise_precision(vec) 
#         self.equ_solver.adjoint_solver(vec)
#         self.p.vector()[:] = self.equ_solver.sol_forward_vec
#         self.q.vector()[:] = self.equ_solver.sol_adjoint_vec
#         b_ = fe.assemble(fe.inner(fe.grad(self.q), self.equ_solver.exp_m*fe.grad(self.p)*self.v_)*fe.dx)
#         return spsl.spsolve(self.equ_solver.M, b_[:])
        
#     def eval_hessian_res_vec(self, m_hat_vec):
#         # self.m_hat.vector()[:] = m_hat_vec
#         self.equ_solver.incremental_forward_solver(m_hat_vec)
#         vec = self.S@self.equ_solver.sol_incremental_vec
#         vec = self._time_noise_precision(vec)        
#         self.equ_solver.incremental_adjoint_solver(vec, m_hat_vec, simple=False)
#         self.pp.vector()[:] = self.equ_solver.sol_incremental_vec
#         self.qq.vector()[:] = self.equ_solver.sol_incremental_adjoint_vec
#         A1 = fe.assemble(fe.inner(self.m_hat*self.equ_solver.exp_m*fe.grad(self.p)*self.v_, \
#                                   fe.grad(self.q))*fe.dx)
#         A2 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.p)*self.v_, 
#                                   fe.grad(self.qq))*fe.dx)
#                          # fe.grad(self.equ_solver.sol_incremental_adjoint))*fe.dx)
#         A3 = fe.assemble(fe.inner(self.equ_solver.exp_m*fe.grad(self.q)*self.v_,
#                                   fe.grad(self.pp))*fe.dx)
#                          # fe.grad(self.equ_solver.sol_incremental))*fe.dx)
        
#         A = A1[:] + A2[:] + A3[:]
        
#         return spsl.spsolve(self.equ_solver.M, A)
    
#     def loss(self):
#         loss_res = self.loss_residual()
#         loss_prior = 0.0
#         return loss_res + loss_prior, loss_res, loss_prior

#     def gradient(self, m_vec):
#         grad_res = self.eval_grad_residual(m_vec)
#         grad_prior = 0.0
#         return grad_res + grad_prior, grad_res, grad_prior

#     def hessian(self, m_vec):
#         hessian_res = self.eval_hessian_res_vec(m_vec)
#         hessian_prior = 0.0
#         return hessian_res + hessian_prior


            
def relative_error(u, v, domain):
    u = fe.interpolate(u, domain.function_space)
    v = fe.interpolate(v, domain.function_space)
    fenzi = fe.assemble(fe.inner(u-v, u-v)*fe.dx)
    fenmu = fe.assemble(fe.inner(v, v)*fe.dx)
    return fenzi/fenmu






