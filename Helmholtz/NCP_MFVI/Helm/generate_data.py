#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:30:05 2023

@author: ishihara
"""

import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# os.chdir("/home/ishihara/Desktop/SIMIP202002/aa")

from NCP_MFVI.common_Helm import EquSolver, Domain2DPML
from core.misc import save_expre, generate_points, trans2spnumpy
from core.probability import GaussianElliptic2
from core.model import Domain2D
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt

import fenics as fe
import dolfin as dl



DATA_DIR = './NCP_MFVI/Helm/DATA/'
RESULT_DIR = './NCP_MFVI/Helm/RESULTS/'
result_figs_dir = RESULT_DIR + "Fig/NCPiMFVI/"

noise_levels = [0.01, 0.05, 0.1]

# domain for solving PDE
equ_nx = 50*6
# equ_nx = 500
dPML = 0.1
domainPML = Domain2DPML(
    nx=equ_nx, ny=equ_nx, dPML=dPML, xx=1.0, yy=1.0
    )
domain = Domain2D(
    low_point=[-dPML, -dPML], high_point=[1+dPML, 1+dPML], \
        nx=equ_nx, ny=equ_nx, mesh_type='CG', mesh_order=1
    )
V = domainPML.function_space
VR, VI = domainPML.function_space_R, domainPML.function_space_I

f_expre01 = '0.3*pow(1-3*(x[0]*2-1), 2)*exp(-pow(3*(x[0]*2-1), 2)-pow(3*(x[1]*2-1)+1, 2))'
f_expre02 = '- (0.2*3*(x[0]*2-1) - pow(3*(x[0]*2-1), 3)-pow(3*(x[1]*2-1), 5))'
f_expre03 = '*exp(-pow(3*(x[0]*2-1),2)-pow(3*(x[1]*2-1),2))'
f_expre04 = '- 0.03*exp(-pow(3*(x[0]*2-1)+1, 2)-pow(3*(x[1]*2-1),2))'
f_expre0 = f_expre01 + f_expre02 + f_expre03 + f_expre04
fR = fe.interpolate(fe.Expression(f_expre0, degree=5), VR)
fR_vec = fR.vector()[:]
fI = fe.interpolate(fe.Expression('0.', degree=3), VI)
fI_vec = fI.vector()[:]

# From the real and imaginary part to rebuild f,
# we rewrite gene_b and adj_solver to simplify the variate from two to one,
# which is f.
f_vec = (np.row_stack((fR_vec, fI_vec)).T).reshape(2 * len(fR_vec))

f = fe.Function(V)
f.vector()[:] = f_vec

f = fR

# fR_init = fe.interpolate(fe.Expression('0.7 * ((pow(x[0] - 1 , 2) + pow(x[1] - 1, 2)))', degree=5), Vv)
# fR_init = fe.interpolate(fe.Expression('0', degree=5), VR)
# fR_init_vec = fR_init.vector()[:]
# f_init = (np.row_stack((fR_init_vec, fI_vec)).T).reshape(2 * len(fI_vec))


os.makedirs(DATA_DIR, exist_ok=True)
np.save(DATA_DIR + 'truth_vec_full', f.vector()[:])

## save the background truth(real part) and mesh
file1 = fe.File(DATA_DIR + "truth_fun.xml")
file1 << fR
file2 = fe.File(DATA_DIR + 'saved_mesh_truth.xml')
file2 << domain.function_space.mesh()


freqs = np.linspace(0.5, 10, 20)*np.pi
np.save(DATA_DIR + "freqs", freqs)

# freq = 1*np.pi
# equ_solver = EquSolver(
#     domainPML=domainPML, m=f, freq=freqs[0], num_points=20
#     )

equ_solver = EquSolver(
    domainPML=domainPML, domain=domain, m=f, freq=freqs[0], num_points=20
)


# b = equ_solver.gene_b(f.vector()[:])
# tmp1, tmp2 = equ_solver.A1_fwd, equ_solver.A2_fwd
# A = tmp1 + freq * freq * tmp2
# soll = spsl.spsolve(A, b)


dcs = []
for idxF, freq in enumerate(freqs):


    equ_solver.update_freq(freq)
    sol = equ_solver.forward_solver(f.vector()[:])
    print("freq = ", equ_solver.freq)
    clean_data = equ_solver.S @ sol
    dcs.append(clean_data)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(sol)
    plt.title('sol')
    plt.subplot(1,2,2)
    plt.plot(clean_data)
    plt.title('clean_data')
    plt.show()
    
np.save(DATA_DIR + "dc", dcs)
# np.save(DATA_DIR + "dc", dcob)





for noise_level in noise_levels:

    ds, taus = [], []

    for idxF, freq in enumerate(freqs):
        np.random.seed(5)
        clean_data = dcs[idxF]
        data_max = max(np.abs(clean_data))
        data = clean_data + noise_level*data_max*np.random.normal(0, 1, len(clean_data))
        ds.append(data)
        taus.append(noise_level*data_max)

    np.save(
        DATA_DIR + "dn" + "_" + str(noise_level), ds
        )
    np.save(
        DATA_DIR + "tau" + "_" + str(noise_level), taus
        )


    print(noise_level)
    

# equ_solver.update_freq(freq)
# sol = equ_solver.forward_solver(f.vector()[:])
# clean_data = equ_solver.Shybird @ sol
# np.save(DATA_DIR + "dc", clean_data)


# plt.figure(figsize=(12, 5))
# plt.subplot(1,2,1)
# plt.plot(sol)
# plt.title('sol')
# plt.subplot(1,2,2)
# plt.plot(clean_data)
# plt.title('clean_data')
# plt.show()


# for noise_level in noise_levels:

#     data_max = max(np.abs(clean_data))
#     data = clean_data + noise_level*data_max * \
#         np.random.normal(0, 1, len(clean_data))
#     tau = noise_level*data_max

#     np.save(
#         DATA_DIR + "dn" + "_" + str(noise_level), data
#     )
#     np.save(
#         DATA_DIR + "tau" + "_" + str(noise_level), tau
#     )

# plt.plot(domainPML.F @ f.vector()[:])











# def gene_b(m_vec):

#     V = domainPML.function_space
#     u = fe.TrialFunction(V)
#     v = fe.TestFunction(V)
#     uR, uI = fe.split(u)
#     vR, vI = fe.split(v)

#     f = fe.Function(V)
#     fR, fI = f.split(deepcopy=True)
#     fR.vector()[:] = m_vec[::2]
#     fI.vector()[:] = m_vec[1::2]

#     b = fe.assemble(fR * vR * fe.dx + fI * vI * fe.dx)

#     def boundary(x, on_boundary):
#         return on_boundary

#     bc = [fe.DirichletBC(V.sub(0), fe.Constant(0), boundary),
#           fe.DirichletBC(V.sub(1), fe.Constant(0), boundary)]

#     # bc[0].apply(b)
#     # bc[1].apply(b)
#     b = b[:]

#     b = b[:]

#     return np.array(b)


# def gene_bcmat():

#     V = domainPML.function_space
#     u = fe.TrialFunction(V)
#     v = fe.TestFunction(V)
#     uR, uI = fe.split(u)
#     vR, vI = fe.split(v)

#     def boundary(x, on_boundary):
#         return on_boundary

#     bc = [fe.DirichletBC(V.sub(0), fe.Constant(0), boundary),
#           fe.DirichletBC(V.sub(1), fe.Constant(0), boundary)]

#     L = fe.inner(fe.Constant(1.0), vR) * fe.dx + \
#         fe.inner(fe.Constant(1.0), vI) * fe.dx
#     b = fe.assemble(L)
#     b_vec0 = np.array(b[:])

#     def boundary(x, on_boundary):
#         return on_boundary

#     bc[0].apply(b)
#     bc[1].apply(b)
#     b_vec = b[:]

#     bc_idx = (b_vec != b_vec0)
#     ans = np.ones(len(b_vec0))
#     ans[bc_idx] = 0.0
#     # for i in range(len(b_vec0)):
#     #     if b_vec0[i] == b_vec[i]:
#     #         ans[i] = 1
#     #     else:
#     #         ans[i] = 0
#     # self.I_bc_vector = ans
#     F = sps.diags(ans)

#     # bc[0].apply(b)
#     # bc[1].apply(b)
#     # b = b[:]

#     return F


# def gene_M():

#     V = domainPML.function_space
#     u = fe.TrialFunction(V)
#     v = fe.TestFunction(V)
#     uR, uI = fe.split(u)
#     vR, vI = fe.split(v)

#     # M_ = fe.inner(uR, vR) * fe.dx + \
#     #     fe.inner(uI, vI) * fe.dx
#     M_ = fe.inner(u, v) * fe.dx
#     M_ = fe.assemble(M_)
#     M = trans2spnumpy(M_)

#     return M


# bb0 = equ_solver.gene_b(f.vector()[:])

# F = gene_bcmat()

# M = gene_M()

# bb = gene_b(f.vector()[:])

# bb1 = F @ bb

# ## matrix formulation
# bb_ = equ_solver.M @ f.vector()[:]
# bb1_ = F @ bb_

# plt.plot(bb0)

