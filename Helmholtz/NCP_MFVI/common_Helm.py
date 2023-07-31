#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:03:37 2023

@author: ishihara
"""

import numpy as np
# import scipy.linalg as sl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import fenics as fe
# import dolfin as dl 

import sys, os
sys.path.append(os.pardir)
from core.model import ModelBase
from core.misc import construct_measurement_matrix, trans2spnumpy
from core.linear_eq_solver import cg_my
from core.approximate_sample import LaplaceApproximate
from core.optimizer import NewtonCG
from core.noise import NoiseGaussianIID


#########################################################################

class Domain2DPML(object):
    def __init__(self, nx=50, ny=50, dPML=0.1, xx=1.0, yy=1.0):
        self.mesh = fe.RectangleMesh(fe.Point(-dPML, -dPML),
                                     fe.Point(dPML + xx, dPML + yy), nx, ny)
        self.P1 = fe.FiniteElement('P', fe.triangle, 1)
        element = fe.MixedElement([self.P1, self.P1])
        self.function_space = fe.FunctionSpace(self.mesh, element)
        self.function_space_R = self.function_space.sub(0).collapse()
        self.function_space_I = self.function_space.sub(1).collapse()
        self.xx, self.yy = xx, yy
        self.dPML = dPML
        self.param = self.parameters()
        self.gene_vector_for_add_bc_I_to_vector()

    def parameters(self):
        xx, yy = self.xx, self.yy
        dPML = self.dPML
        sig0 = 1.5
        p = 2.3
        sig1 = fe.Expression('x[0] > x2 && x[0] < x2+dd ? \
                              sig0_*pow((x[0]-x2)/dd, p_) : \
                (x[0] < x1 && x[0] > x1-dd ? sig0_*pow((x1-x[0])/dd, p_) : 0)',
                              degree=3, x1=0, x2=xx, dd=dPML, sig0_=sig0, p_=p)
        sig2 = fe.Expression('x[1] > y2 && x[1] < y2+dd ? \
                              sig0_*pow((x[1]-y2)/dd, p_) : \
                (x[1] < y1 && x[1] > y1-dd ? sig0_*pow((y1-x[1])/dd, p_) : 0)',
                              degree=3, y1=0, y2=yy, dd=dPML, sig0_=sig0, p_=p)
        
            
        sR = fe.as_matrix([[(1+sig1*sig2)/(1+sig1*sig1), 0.0],
                           [0.0, (1+sig1*sig2)/(1+sig2*sig2)]])
        sI = fe.as_matrix([[(sig2-sig1)/(1+sig1*sig1), 0.0],
                           [0.0, (sig1-sig2)/(1+sig2*sig2)]])
        # here we set p(x, y) = sig1(x) *  sig2(y)
        pR = 1 - sig1 * sig2
        pI = sig1 + sig2
        param = [sR, sI, pR, pI]
        return param

    def gene_vector_for_add_bc_I_to_vector(self):
        '''
        Add Dirichlet boundary condition to a np vector
        '''
        v = fe.TestFunction(self.function_space)
        vR, vI = fe.split(v)
        L = fe.inner(fe.Constant(1.0), vR) * fe.dx + \
            fe.inner(fe.Constant(1.0), vI) * fe.dx
        b = fe.assemble(L)
        b_vec0 = np.array(b[:])

        def boundary(x, on_boundary):
            return on_boundary

        bc = [fe.DirichletBC(
            self.function_space.sub(0), fe.Constant(0), boundary),
            fe.DirichletBC(
            self.function_space.sub(1), fe.Constant(0), boundary)]
        bc[0].apply(b)
        bc[1].apply(b)
        self.b_vec = b[:]
        
        self.bc_idx = (self.b_vec != b_vec0)
        self.ans = np.ones(len(b_vec0))
        self.ans[self.bc_idx] = 0.0
        # for i in range(len(b_vec0)):
        #     if b_vec0[i] == b_vec[i]:
        #         ans[i] = 1
        #     else:
        #         ans[i] = 0
        # self.I_bc_vector = ans
        self.F = sps.diags(self.ans)
        
        
        
        
class EquSolver(object):
    def __init__(
            self, domainPML, domain, m=None, freq=None, num_points=40
    ):

        self.domainPML = domainPML
        self.domain = domain
        self.paras = self.domainPML.param

        # if m is None:
        #     self.m = fe.interpolate(fe.Constant("0.0"), self.domain.function_space)
        # else:
        #     self.m = fe.interpolate(m, self.domainPML.function_space)
        
        self.points = num_points
        self.freq = freq
        self.update_points(dim=self.points)
        # self.update_freq(self.freq)
        self.m = fe.Function(self.domain.function_space)
        # self.m_vec = self.m.vector()[:]
        # self.q_fun = fe.Expression('1.', degree=3)
        self._gene_fwd_matrix()
        self._gene_adj_matrix()
        # self.construct_matrix()
        
        u_ = fe.TrialFunction(self.domainPML.function_space)
        v_ = fe.TestFunction(self.domainPML.function_space)
        M_ = fe.assemble(fe.inner(u_, v_) * fe.dx)
        self.M = trans2spnumpy(M_)
        
        u_ = fe.TrialFunction(self.domain.function_space)
        v_ = fe.TestFunction(self.domain.function_space)
        M_ = fe.assemble(fe.inner(u_, v_) * fe.dx)
        self.MM = trans2spnumpy(M_)
        
        self.F = self.domainPML.F
        
        # m_vec_1 = fe.interpolate(fe.Constant("0.0"), self.domainPML.function_space_R)
        # self.vec1 = np.array(m_vec_1.vector()[:])
        m_vec_2 = fe.interpolate(fe.Constant("0.0"), self.domainPML.function_space_I)
        self.vec2 = 0 * np.array(m_vec_2.vector()[:])
        
        self.leng = len(np.array(self.vec2).squeeze())
        
        # self.vec = (np.row_stack((self.vec1, self.vec2)).T).reshape(2*self.leng)
        
    
    def construct_full_vector(self, m_vec):
        
        self.vec1 = np.array(m_vec[:]).squeeze()
        
        if len(self.vec1.shape) == 1:
            
            m_vec_2 = fe.interpolate(fe.Constant("0.0"), self.domainPML.function_space_I)
            self.vec2 = 0 * np.array(m_vec_2.vector()[:])
            self.vec = (np.row_stack((self.vec1, self.vec2)).T).reshape(2*self.leng)
        
        else:
            
            self.vec2 = self.vec1 * 0
            
            x, y = self.vec1.shape
            cc = np.concatenate((self.vec1, self.vec2), axis=1)
            cc = np.reshape(cc, (x * 2, y))
            self.vec = cc
        
    def update_m(self, m_vec):
        assert len(self.m.vector()[:]) == len(m_vec)
        self.construct_full_vector(m_vec)
        self.m.vector()[:] = np.array(m_vec[:])
        # self.m.vector()[:] = np.array(self.vec[:])
        
    def update_freq(self, freq):
        self.freq = freq
    
    def update_points(self, points=None, dim=40):
        if points is not None:
            self.points = points
        if dim is not None:
            x_vec = np.linspace(0, self.domainPML.xx, dim)
            y_vec = np.linspace(0, self.domainPML.yy, dim)
            points = []
            for y in y_vec:
                points.append((x_vec[0], y))
            for i in range(1, len(x_vec) - 1):
                points.append((x_vec[i], y_vec[0]))
                points.append((x_vec[i], y_vec[-1]))
            for y in y_vec:
                points.append((x_vec[-1], y))
        self.points = np.array(points)
        
        self.gene_Sample_matrix()
    
    def gene_Sample_matrix(self):
        # self.update_points()
        SR = construct_measurement_matrix(self.points, self.domainPML.function_space_R)
        SI = construct_measurement_matrix(self.points, self.domainPML.function_space_I)
    
        xxx, yyy = SR.shape
        Ssupp = np.zeros((xxx, yyy))
    
        def gene_hybrid_matrix(p, q):
            x, y = p.shape
            p_ = np.reshape(p, (1, x, y))
            q_ = np.reshape(q, (1, x, y))
            c = np.concatenate((p_, q_), axis=0)
            c = np.transpose(c, (1, 2, 0))
            c = np.reshape(c, (x, y * 2))
            return c
    
        def gene_hybrid_S(p, q):
            x, y = p.shape
            cc = np.concatenate((p, q), axis=1)
            cc = np.reshape(cc, (x * 2, y))
            return cc
    
        SR_hybrid = gene_hybrid_matrix(SR.toarray()[:], Ssupp)
        SI_hybrid = gene_hybrid_matrix(Ssupp, SI.toarray()[:])
        Shybrid = gene_hybrid_S(SR_hybrid, SI_hybrid)
        self.Shybird = Shybrid
        self.S = self.Shybird

      
        
    def _gene_fwd_matrix(self):
        
        sR, sI = self.paras[0], self.paras[1]
        pR, pI = self.paras[2], self.paras[3]

        
        u = fe.TrialFunction(self.domainPML.function_space)
        v = fe.TestFunction(self.domainPML.function_space)
        
        uR, uI = fe.split(u)
        vR, vI = fe.split(v)

        
        def sR_(p):
            return fe.dot(sR, fe.grad(p))
        def sI_(p):
            return fe.dot(sI, fe.grad(p))
        
        A1_ = -fe.inner(sR_(uR) - sI_(uI), fe.grad(vR)) * fe.dx - \
            fe.inner(sR_(uI) + sI_(uR), fe.grad(vI)) * fe.dx 
        
        A1_ = fe.assemble(A1_)
        
        
        A2_ = fe.inner(pR * uI + pI* uR, vI) * fe.dx + \
            fe.inner(pR * uR - pI * uI, vR) * fe.dx   
            
            
        A2_ = fe.assemble(A2_)

        
        # define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary
        
        self.bc = [fe.DirichletBC(self.domainPML.function_space.sub(0), fe.Constant(0), boundary),
              fe.DirichletBC(self.domainPML.function_space.sub(1), fe.Constant(0), boundary)]

        self.bc[0].apply(A1_)
        self.bc[1].apply(A1_)
        self.bc[0].apply(A2_)
        self.bc[1].apply(A2_)
        
        self.A1_fwd = trans2spnumpy(A1_)
        self.A2_fwd = trans2spnumpy(A2_)
        
        
    def _gene_adj_matrix(self):
        
        sR, sI = self.paras[0], self.paras[1]
        pR, pI = self.paras[2], self.paras[3]

        
        u = fe.TrialFunction(self.domainPML.function_space)
        v = fe.TestFunction(self.domainPML.function_space)
        
        uR, uI = fe.split(u)
        vR, vI = fe.split(v)

        
        def sR_(p):
            return fe.dot(sR, fe.grad(p))
        def sI_(p):
            return fe.dot(sI, fe.grad(p))
        
        A1_ = -fe.inner(sR_(uR) + sI_(uI), fe.grad(vR)) * fe.dx - \
            fe.inner(sR_(uI) - sI_(uR), fe.grad(vI)) * fe.dx

        A2_ = fe.inner(pR * uI - pI * uR, vI) * fe.dx + \
            fe.inner(pR * uR + pI * uI, vR) * fe.dx
        
            
        A1_ = fe.assemble(A1_)
            
        A2_ = fe.assemble(A2_)
        
        def boundary(x, on_boundary):
            return on_boundary

        self.bc[0].apply(A1_)
        self.bc[1].apply(A1_)
        self.bc[0].apply(A2_)
        self.bc[1].apply(A2_)

        self.A1_adj = trans2spnumpy(A1_)
        self.A2_adj = trans2spnumpy(A2_)

            
            
    # def gene_b(self, m_vec):

        
    #     V = self.domainPML.function_space
    #     u = fe.TrialFunction(V)
    #     v = fe.TestFunction(V)
    #     uR, uI = fe.split(u)
    #     vR, vI = fe.split(v)
        

    #     f = fe.Function(V)
    #     fR, fI = f.split(deepcopy=True)
    #     fR.vector()[:] = m_vec[::2]
    #     fI.vector()[:] =  m_vec[1::2]
        
        
    #     self.b = fe.assemble(fR * vR * fe.dx + fI * vI * fe.dx)
        
    #     def boundary(x, on_boundary):
    #         return on_boundary

    #     self.bc[0].apply(self.b)
    #     self.bc[1].apply(self.b)
    #     b = self.b[:]
        
    #     return np.array(b)
    
    # def gene_b_(self, m_vec):

        
    #     V = self.domainPML.function_space
    #     u = fe.TrialFunction(V)
    #     v = fe.TestFunction(V)
    #     uR, uI = fe.split(u)
    #     vR, vI = fe.split(v)
        

    #     f = fe.Function(V)
    #     fR, fI = f.split(deepcopy=True)
    #     fR.vector()[:] = m_vec[::2]
    #     fI.vector()[:] =  m_vec[1::2]
        
        
    #     b = fe.assemble(fR * vR * fe.dx + fI * vI * fe.dx)
        
    #     b = b[:]
        
    #     return np.array(b)
    
    
    
    def forward_solver(self, m_vec=None):
        
        if m_vec is not None:
            self.update_m(m_vec)

        A = self.A1_fwd + self.freq*self.freq*self.A2_fwd
        # A = self.A_fwd
        
        F = self.F @ self.M
        self.b = F @ self.vec[:]
        # m_vec[self.domainPML.bc_idx] = 0.0
        # self.b = m_vec
        # self.b = self.gene_b(m_vec)

        self.forward_sol = spsl.spsolve(A, self.b)
        self.forward_sol = np.array(self.forward_sol)
        
        return np.array(self.forward_sol)
    
    def incremental_forward_solver(self, m_hat=None):
        ## we need this function can accept matrix input
        ## For linear problems, the incremental forward == forward
        
        # self._gene_fwd_martix()
        
        A = self.A1_fwd + self.freq*self.freq*self.A2_fwd
        # A = self.A_fwd
        
        F = self.F @ self.M
        
        self.construct_full_vector(m_hat)
        self.b = F @ self.vec
        
        # self.b = F @ m_hat
        
        # self.b = self.gene_b(m_hat)
        
        self.inc_forward_sol = np.array(spsl.spsolve(A, self.b))
        
        return self.inc_forward_sol
    
    def adjoint_solver(self, res_vec):
        
        # self._gene_adj_martix()
        
        A = self.A1_adj + self.freq*self.freq*self.A2_adj
        # A = self.A_adj
        
        # rhs = self.S.T @ res_vec
        # self.sol_adjoint_vec = self.F.T @ A.T @ rhs
        
        # rhs = -1 * self.S.T @ res_vec
        rhs = self.S.T @ res_vec
        # rhs = self.gene_b(rhs)
        # tmp = spsl.spsolve(A, rhs)
        
        tmp = self.F @ spsl.spsolve(A, rhs)
        
        self.sol_adjoint_vec = tmp
        
        self.adjoint_sol = np.array(self.sol_adjoint_vec)[::2]
        
        return self.adjoint_sol
        
    
    def incremental_adjoint_solver(self, vec, m_hat=None):   
        
        self.inc_adjoint_sol = self.adjoint_solver(vec)
        self.inc_adjoint_sol = np.array(self.inc_adjoint_sol)
        
        return self.inc_adjoint_sol
        

        
#######################################################################


class GaussianLam(object):
    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov
        
    def generate_sample(self):
        return np.random.normal(self.mean, np.sqrt(self.cov))

class ModelHelmNCP(ModelBase):
    
    def __init__(self, ds, domain, prior, noise, equ_solver, lam_dis, freq, tau):
        super().__init__(ds, domain, prior, noise, equ_solver)
        
        self.ds = ds
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        self.freq = freq
        self.tau = tau
        self.num = 0
        
        self.update_noise(self.tau)
        self.update_d(self.ds)
        self.equ_solver.update_freq(self.freq)
        
        
    def update_noise(self, tau):
        
        if tau is None:
            self.tau = tau
        self.noise.set_parameters(variance=tau**2)    
            
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.mean
            
    def update_m(self, m_vec, update_sol=True):
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
            
            
    def update_d(self, dd):
        self.ds = dd
            
        
    def update_paras(self, freq, ds, tau):
        self.equ_solver.update_freq(freq)
        self.freq = freq
        self.ds = ds
        self.tau = tau
        self.noise.set_parameters(variance=self.tau**2)    
        
    def loss_residual(self, m_vec=None):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
    
        self.update_noise(self.tau)
        self.equ_solver.update_freq(self.equ_solver.freq)
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.noise.mean - self.ds)
        if self.noise.precision is None:
            temp = temp@spsl.spsolve(self.noise.covariance, temp)
        else:
            temp = temp@self.noise.precision@temp
        tmp = temp
        
        # tmp = 0
        
        # for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
        #     self.update_noise(self.taus(ii))
        #     self.equ_solver.update_freq(freq)
        #     self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        #     temp = cc*(self.S@self.p.vector()[:])
        #     temp = (temp - self.noise.mean - self.ds[ii])
        #     if self.noise.precision is None:
        #         temp = temp@spsl.spsolve(self.noise.covariance, temp)
        #     else:
        #         temp = temp@self.noise.precision@temp
        #     tmp += temp
            
        return 0.5*tmp
            
    
    def loss_residual_L2(self, m_vec=None):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        self.update_noise(self.tau)
        self.equ_solver.update_freq(self.equ_solver.freq)
        self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        temp = cc*(self.S@self.p.vector()[:])
        temp = (temp - self.noise.mean - self.ds)

        tmp = temp @ temp
        # tmp = 0
        
        # for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
        #     self.equ_solver.update_freq(freq)
        #     self.update_noise(self.taus[ii])
            
        #     self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        #     ## scale lam
        #     measure_scale = cc*self.S@(self.p.vector()[:])
        #     temp = measure_scale - self.d
        #     temp = temp @ temp
        #     tmp += temp
        
        return 0.5*tmp
                
    def eval_grad_residual(self, m_vec):
        
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        # cc=1
        
        self.equ_solver.update_freq(self.equ_solver.freq)
        self.update_noise(self.tau)
        
        tmp = cc * self.equ_solver.S @ self.equ_solver.forward_solver(m_vec)
        # tmp = -1 * (tmp - dd)
        tmp = spsl.spsolve(self.noise.covariance, tmp - self.ds)
       
        g = self.equ_solver.adjoint_solver(tmp)
        
        
        # tmp = 0
        
        # for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
        #     self.equ_solver.update_freq(freq)
        #     self.update_noise(self.taus[ii])
            
        #     self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        #     ## scale lam
        #     measure_scale = cc*self.S@(self.p.vector()[:])
        #     res_vec = spsl.spsolve(self.noise.covariance, measure_scale - self.ds[ii])
        #     # print(res_vec)
        #     self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        #     g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        #     tmp += g_.vector()[:]
            ## scale lam
        # g = cc*g_.vector()[:]
        
        g = cc*g
        return np.array(g)
    
    def eval_hessian_res_vec(self, dm):
        
        
        
        # cc=1
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        self.equ_solver.update_freq(self.equ_solver.freq)
        self.update_noise(self.tau)
        
        tmp = cc*self.equ_solver.S @ self.equ_solver.forward_solver(dm)
        # tmp = -1 * (tmp - dd)
        tmp = spsl.spsolve(self.noise.covariance, tmp)
       
        self.q = self.equ_solver.adjoint_solver(tmp)
        HM = self.q
        
        
        # self.equ_solver.update_freq(self.equ_solver.freq)
        # self.update_noise(self.tau)
    
        # self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        # measure = (self.S@(self.p.vector()[:]))
        # res_vec = spsl.spsolve(self.noise.covariance, measure)
        # self.q = self.equ_solver.adjoint_solver(res_vec)
        # # g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        # # HM = g_.vector()[:]
        # HM = self.q
        
        # temp = 0
        # for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
        #     self.equ_solver.update_m(dm)
        #     self.equ_solver.update_freq(freq)
        #     self.update_noise(self.taus[ii])
            
        #     self.p.vector()[:] = self.equ_solver.forward_solver(dm)
        #     measure = (self.Shybird@(self.p.vector()[:]))
        #     res_vec = spsl.spsolve(self.noise.covariance, measure)
        #     self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
        #     g_ = fe.interpolate(self.q, self.domain_equ.function_space)
        #     HM = g_.vector()[:]
        #     temp += HM

        # tmp = self.lam*self.lam + self.lam_dis.cov
        return cc*np.array(HM)
    
    def eval_HAdjointData(self):
        
        temp = 0
        for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
            self.update_noise(self.taus[ii])
            
            res_vec = spsl.spsolve(self.noise.covariance, self.ds[ii])
            self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
            g_ = fe.interpolate(self.q, self.domain_equ.function_space)
            ## scale lam
            g = self.lam*g_.vector()[:]
            temp += g
        # return np.array(g)
        return np.array(temp)
    
    def update_lam_dis(self, m, eigval):
        tmp = self.lam_dis.cov + self.lam*self.lam
        eigval = eigval/tmp
        # update covariance
        
        tmpp = 0
        for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
            self.equ_solver.update_freq(freq)
            self.update_noise(self.taus[ii])
            
            self.p.vector()[:] = self.equ_solver.forward_solver(m)
            p = self.p.vector()[:]
            Sp = self.S@p
            if self.noise.precision is None:
                temp = Sp@spsl.spsolve(self.noise.covariance, Sp)
            else:
                temp = Sp@self.noise.precision@Sp
            tmpp += temp    
                
        # temp1 = temp + 1/self.lam_cov0
        temp1 = tmpp + 1/self.lam_cov0
        rho = self.lam_dis.cov + self.lam*self.lam
        temp2 = np.sum(eigval/(rho*eigval + 1))
        self.lam_dis.cov = 1/(temp1 + temp2)
        # print("-----", temp1, temp2)
        
        # update mean
        if self.noise.precision is None:
            tmp = self.ds[ii] @ spsl.spsolve(self.noise.covariance, Sp)
        else:
            tmp = self.ds[ii] @ self.noise.precision @ (Sp)
        # print("*****", tmp, temp)
            
        self.lam_dis.mean = self.lam_dis.cov*(tmp + self.lam_mean0/self.lam_cov0)
        # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
    
        
    
class ModelHelmNCP_Mult(ModelBase):
    
    def __init__(self, ds, domain, prior, noise, equ_solver, lam_dis, freqs, taus):
        super().__init__(ds, domain, prior, noise, equ_solver)
        
        self.ds = ds
        self.lam_dis = lam_dis
        self.lam_mean0 = self.lam_dis.mean
        self.lam_cov0 = self.lam_dis.cov
        self.lam = self.lam_dis.mean
        self.freqs = freqs
        self.taus = taus
        self.start = 0
        self.end = len(self.freqs)
        
        self.update_noise(self.taus[0])
        self.update_d(self.ds[0])
        self.equ_solver.update_freq(self.freqs[0])
        
        
    def update_noise(self, tau):
        
        if tau is None:
            self.tau = tau
        self.noise.set_parameters(variance=tau**2)    
            
    def update_lam(self, lam=None):
        if lam is not None:
            self.lam = lam
        else:
            self.lam = self.lam_dis.mean
            
    def update_m(self, m_vec, update_sol=True):
        self.m.vector()[:] = np.array(m_vec)
        self.equ_solver.update_m(m_vec)
        if update_sol == True:
            self.equ_solver.forward_solver()
            
            
    def update_d(self, dd):
        self.d = dd
            
        
    def update_paras(self, freq, ds, tau):
        self.equ_solver.update_freq(freq)
        self.freq = freq
        self.d = ds
        self.tau = tau
        self.noise.set_parameters(variance=self.tau**2)    
        
    def loss_residual(self, m_vec=None):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
    
        # self.update_noise(self.tau)
        # self.equ_solver.update_freq(self.equ_solver.freq)
        # self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        # temp = cc*(self.S@self.p.vector()[:])
        # temp = (temp - self.noise.mean - self.ds)
        # if self.noise.precision is None:
        #     temp = temp@spsl.spsolve(self.noise.covariance, temp)
        # else:
        #     temp = temp@self.noise.precision@temp
        # tmp = temp
        
        tmp = 0
        
        for ii, freq in list(enumerate(self.freqs))[self.start: self.end]:
            
            self.update_noise(self.taus[ii])
            self.equ_solver.update_freq(freq)
            self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
            temp = cc*(self.S@self.p.vector()[:])
            temp = (temp - self.noise.mean - self.ds[ii])
            if self.noise.precision is None:
                temp = temp@spsl.spsolve(self.noise.covariance, temp)
            else:
                temp = temp@self.noise.precision@temp
            tmp += temp
            
        return 0.5*tmp
            
    
    def loss_residual_L2(self, m_vec=None):
        ## scale lam
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        # self.update_noise(self.tau)
        # self.equ_solver.update_freq(self.equ_solver.freq)
        # self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
        # temp = cc*(self.S@self.p.vector()[:])
        # temp = (temp - self.noise.mean - self.ds)

        # tmp = temp @ temp
        
        tmp = 0
        
        for ii, freq in list(enumerate(self.freqs))[self.start: self.end]:
            
            
            self.update_noise(self.taus[ii])
            self.equ_solver.update_freq(self.freq)
            self.p.vector()[:] = self.equ_solver.forward_solver(m_vec)
            temp = cc*(self.S@self.p.vector()[:])
            temp = (temp - self.noise.mean - self.ds[ii])
            

            temp = temp @ temp
            tmp += temp
        
        return 0.5*tmp
                
    def eval_grad_residual(self, m_vec):
        
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        # cc=1
        
        # self.equ_solver.update_freq(self.equ_solver.freq)
        # self.update_noise(self.tau)
        
        # tmp = cc * self.equ_solver.S @ self.equ_solver.forward_solver(m_vec)
        # # tmp = -1 * (tmp - dd)
        # tmp = spsl.spsolve(self.noise.covariance, tmp - self.ds)
       
        # g = self.equ_solver.adjoint_solver(tmp)
        
        
        g = 0
        
        for ii, freq in list(enumerate(self.freqs))[self.start: self.end]:
            
            self.equ_solver.update_freq(freq)
            self.update_noise(self.taus[ii])
            
            temp = cc * self.equ_solver.S @ self.equ_solver.forward_solver(m_vec)
            temp = spsl.spsolve(self.noise.covariance, temp - self.ds[ii])
            
            temp = self.equ_solver.adjoint_solver(temp)
            g += temp

        
        g = cc*g
        return np.array(g)
    
    def eval_hessian_res_vec(self, dm):
        
        
        
        # cc=1
        cc = np.sqrt(self.lam*self.lam + self.lam_dis.cov)
        # self.equ_solver.update_freq(self.equ_solver.freq)
        # self.update_noise(self.tau)
        
        # tmp = cc*self.equ_solver.S @ self.equ_solver.forward_solver(dm)
        # # tmp = -1 * (tmp - dd)
        # tmp = spsl.spsolve(self.noise.covariance, tmp)
       
        # self.q = self.equ_solver.adjoint_solver(tmp)
        # HM = self.q
        
        
        HM = 0
        for ii, freq in list(enumerate(self.freqs))[self.start: self.end]:
            
            self.equ_solver.update_freq(self.equ_solver.freq)
            self.update_noise(self.taus[ii])
            
            tmp = cc*self.equ_solver.S @ self.equ_solver.forward_solver(dm)
            # tmp = -1 * (tmp - dd)
            tmp = spsl.spsolve(self.noise.covariance, tmp)
           
            self.q = self.equ_solver.adjoint_solver(tmp)
            HM += self.q

        # tmp = self.lam*self.lam + self.lam_dis.cov
        return cc*np.array(HM)
    
    def eval_HAdjointData(self):
        
        temp = 0
        for ii, freq in list(enumerate(self.freqs))[self.num:]:
            
            self.update_noise(self.taus[ii])
            
            res_vec = spsl.spsolve(self.noise.covariance, self.ds[ii])
            self.q.vector()[:] = self.equ_solver.adjoint_solver(res_vec)
            g_ = fe.interpolate(self.q, self.domain_equ.function_space)
            ## scale lam
            g = self.lam*g_.vector()[:]
            temp += g
        # return np.array(g)
        return np.array(temp)
    
    def update_lam_dis(self, m, eigval):
        tmp = self.lam_dis.cov + self.lam*self.lam
        eigval = eigval/tmp
        # update covariance
        
        rho = self.lam_dis.mean*self.lam_dis.mean + self.lam_dis.cov
        vk = m * self.lam_dis.mean / np.sqrt(rho)
        
        
        # tmp = lam_dis.cov + lam_dis.mean * lam_dis.mean
        # eigval = eigval/tmp
        eig_val = 0
        # update covariance
        
        Hvk2 = 0
        dHvk = 0
        
        for ii, freq in list(enumerate(self.freqs))[self.start: self.end]:
            
            self.equ_solver.update_freq(freq)
            self.update_noise(self.taus[ii])
            
            
            m = vk
            p = self.equ_solver.forward_solver(m)
            Sp = self.S@p
            
            
            if self.noise.precision is None:
                tmp1 = Sp@spsl.spsolve(self.noise.covariance, Sp)
                tmp2 = self.ds[ii] @ spsl.spsolve(self.noise.covariance, Sp)
            else:
                tmp1 = Sp@self.noise.precision@Sp
                tmp2 = self.ds[ii] @ self.noise.precision @ (Sp)
                
                
            Hvk2 += tmp1
            dHvk += tmp2
                
        # temp1 = temp + 1/self.lam_cov0
        temp1 = Hvk2 + 1/self.lam_cov0
        rho = self.lam_dis.cov + self.lam*self.lam
        temp2 = 0
        # temp2 = np.sum(eigval/(rho*eigval + 1))
        self.lam_dis.cov = 1/(temp1 + temp2)
        
        print("-----", Hvk2, dHvk)
        
        
        
        # update mean
        self.lam_dis.mean = self.lam_dis.cov*(dHvk + self.lam_mean0/self.lam_cov0)
        # print("+++++", self.lam_mean0/self.lam_cov0, tmp/temp)
        
   
    
    
    
    
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
        self.eigval_lam = self.eigval*(self.model.lam*self.model.lam + self.model.lam_dis.cov)
     
    
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
        # newton_cg = NewtonCG(model=self.model)
        
        # ## calculate the posterior mean 
        # max_iter = iter_num
        # loss_pre, _, _ = self.model.loss()
        # for itr in range(max_iter):
        #     newton_cg.descent_direction(cg_max=cg_max, method=method)
        #     newton_cg.step(method='armijo', show_step=False)
        #     loss, _, _ = self.model.loss()
        #     # print("iter = %d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        #     if newton_cg.converged == False:
        #         break
        #     if np.abs(loss - loss_pre) < 1e-5*loss:
        #         # print("Iteration stoped at iter = %d" % itr)
        #         break 
        #     loss_pre = loss
        
        # tmp = self.model.lam/np.sqrt(self.model.lam*self.model.lam + self.model.lam_dis.cov)
        # self.mean = tmp*newton_cg.mk        
        
        
        self.newton_cg.re_init(self.newton_cg.mk.copy())
        loss_pre = self.model.loss()[0]
        max_iter = iter_num
        for itr in range(max_iter):
            
            
            # NewtonCG
            # newton_cg.descent_direction(cg_max=50, method='cgs')
            self.newton_cg.descent_direction(cg_max=50, method='cg_my')
            # print(newton_cg.hessian_terminate_info)
            # gradient_decent.lr = 1e-6
            # gradient_decent.step(method='fixed')
            self.newton_cg.step(method='armijo', show_step=False)
            # if newton_cg.converged == False:
            #     newton_cg.descent_direction(cg_max=0, method='cg_my')
            #     newton_cg.step(method='armijo', show_step=False)
            # if newton_cg.converged == False:
            #     break
            loss = self.model.loss()[0]
            # print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
            # print("loss_res, loss_reg = ", model.loss()[1], model.loss()[2])
            # if np.abs(loss - loss_pre) < 1e-3*loss:
            #     print("Iteration stoped at iter = %d" % itr)
            #     break 
            loss_pre = loss
            
        self.mean = np.array(self.newton_cg.mk.copy())
                
                
                
                
                
                
                
                

