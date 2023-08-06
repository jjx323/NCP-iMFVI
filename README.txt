Code for the paper: 
Jiaming Sui, Junxiong Jia, Non-Certered Parametrization based infinite-dimensional mean-field variational inference approach, 2023, https://arxiv.org/abs/2211.10703

This program is based on the program "https://github.com/jjx323/IPBayesML" and the following articles:

[1] Tan Bui-Thanh and Omar Ghattas, Analysis of the Hessian for inverse scattering problems: I. Inverse shape
scattering of acoustic waves, Inverse Problems 28 (2012), no. 5, 055001.

[2] Umberto Villa, Noemi Petra, and Omar Ghattas, hIPPYlib: An extensible software framework for large-scale
inverse problems governed by PDEs: Part I: Deterministic inversion and linearized Bayesian inference, ACM
Transactions on Mathematical Software (TOMS) 47 (2021), no. 2, 1-34.

[3] Jiaming Sui and Junxiong Jia, Non-centered parametric variational Bayes' approach for hierarchical inverse problems of partial differential equations'' 2023.

[4] Junxiong Jia, Peijun Li, and Deyu Meng, Stein variational gradient descent on infinite-dimensional space and
applications to statistical inverse problems, SIAM Journal on Numerical Analysis 60 (2022), no. 4, 2225-2252.

############################################################################
Requirements and Dependencies:

Deepin 20.2.3, Anaconda 4.10.3
Python 3.9.6
FEniCS 2019, numpy 1.21.2, scipy 1.7.1, matplotlib 3.4.3

############################################################################
NCP-iMFVI method:

In this program, we apply the NCP-iMFVI method to three typical inverse problems, including two linear and one non-linear problems.

In the first case, we consider a simple elliptic equation.
In the second case, we consider a large-scale inverse source problem of Helmholtz equation.
In the third case, we consider a non-linear inverse problem of steady-state Darcy-flow equation.

############################################################################
CASE1: Simple elliptic equation:

Step1: Generate data

	python generate_data.py

The data is generated in /DATA/
*Readers can change the path by changing viariable ``DATA_DIR''

Step2: Solving and Plotting

Employing NCP-iMFVI	python NCPiMFVI.py

Results are generated in /RESULTS/Fig/NCPiMFVI.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

Figures:
Estimate.png is the sub-figure (a) of Figure 2 in [3].
Estimateg.png is the sub-figure (b) of Figure 2 in [3].
Relative.png is the sub-figure (a) of Figure 3 in [3].
lams.png is the sub-figure (b) of Figure 3 in [3].
Covarianceu.png is the sub-figure (a) of Figure 4 in [3].


Employing Gibbs sampling	python Gibbs_sampler.py

Results are generated in /RESULTS/Fig/GibbsSampler.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

Figures:
lams.png is the sub-figure (c) of Figure 2 in [3].
Covarianceu.png is the sub-figure (b) of Figure 4 in [3].
Covariancediff.png is the sub-figure (c) of Figure 4 in [3].
Varianceu.png is the sub-figure (a) of Figure 5 in [3].
Varianceu20.png is the sub-figure (b) of Figure 5 in [3].
Varianceu40.png is the sub-figure (c) of Figure 5 in [3].


Step3: lambda self-adjusted
	
	python lambda_selfadjust.py

Figures:
lam.png is the sub-figure (a) of Figure 7 in [3].
lamhalf.png is the sub-figure (b) of Figure 7 in [3].


Step4: Dimensional Independence

	python meshinde.py

change variable ``equ_nx''={100, 300, 500 ,700 ,900}
load variables ``norm1, norm2, norm3, norm4, norm5''
load function ``plot_''
run ``plot_(1, 51)''

Figure:
inde.png is the sub-figure (b) of Figure 6 in [3].


############################################################################
CASE2: Helmholtz equation:


For this two-dimensional case, to simulate the problem defined on $\mathbb{R}^2$, we use the uniaxial perfectly matched layer (PML)
technique to trubcate the whole space $\mathbb{R}^2$ in to a bounded domain.
Readers can seek more details about PML in the following articles:
[5] Gang Bao, Shui-Nee Chow, Peijun Li, and Haomin Zhou, Numerical solution of an inverse medium scattering
problem with a stochastic source, Inverse Problems 26 (2010), no. 7, 074014.
[6] Junxiong Jia, Bangyu Wu, Jigen Peng, and Jinghuai Gao, Recursive linearization method for inverse medium
scattering problems with complex mixture Gaussian error learning, Inverse Problems 35 (2019), no. 7, 075003.
[7] Junxiong Jia, Yanni Wu, Peijun Li, Deyu Meng, Variational inverting network for statistical inverse problems of partial differential equations, Journal of Machine Learning Research, 24, 1--60, 2023.

Step1: Generate data

	python generate_data.py

The data is generated in /NCP_MFVI/Helm/DATA
*Readers can change the path by changing viariable ``DATA_DIR''

Step2: Solving and Plotting

Employing NCP-iMFVI	python NCPiMFVI.py

Results are generated in /NCP_MFVI/Helm/RESULTS/Fig/NCPiMFVI.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

Figures:
Truth.png is the sub-figure (a) of Figure 8 in [3].
Estimate.png is the sub-figure (b) of Figure 8 in [3].
Pointwise.png is the sub-figure (c) of Figure 8 in [3].
Credibility region.png is the sub-figure (a) of Figure 9 in [3].
Credibility regionsub1.png is the sub-figure (b) of Figure 9 in [3].
Credibility regionsub2.png is the sub-figure (c) of Figure 9 in [3].
Relative.png is the sub-figure (a) of Figure 10 in [3].
lams.png is the sub-figure (b) of Figure 10 in [3].


Step3: Dimensional Independence

	python meshinde.py

Results are generated in /NCP_MFVI/Helm/RESULTS/Fig/NCPiMFVI.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

change variable ``equ_nx''={60, 65, 70 ,75, 80}
load variables ``norm1, norm2, norm3, norm4, norm5''
load function ``plot_''
run ``plot_(0, 15)''

Figure:
inde.png is the sub-figure (c) of Figure 10 in [3].


############################################################################
CASE3: steady-state Darcy-flow equation:


Step1: Generate data

	python generate_data.py

The data is generated in /NCP_MFVI/Helm/DATA
*Readers can change the path by changing viariable ``DATA_DIR''

Step2: Generate MAP points

	python eval_MAP.py

The data is generated in /NCP_MFVI/Darcy/DATA
*Readers can change the path by changing viariable ``DATA_DIR''

Step3: Solving and Plotting

Employing NCP-iMFVI	python NCPiMFVI.py

Results are generated in /NCP_MFVI/Darcy/RESULTS/Fig/NCPiMFVI.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

Figures:
Truth.png is the sub-figure (a) of Figure 11 in [3].
Estimate.png is the sub-figure (b) of Figure 11 in [3].
Pointwise.png is the sub-figure (c) of Figure 11 in [3].
Credibility region.png is the sub-figure (a) of Figure 12 in [3].
Credibility regionsub1.png is the sub-figure (b) of Figure 12 in [3].
Credibility regionsub2.png is the sub-figure (c) of Figure 12 in [3].
Relative.png is the sub-figure (a) of Figure 13 in [3].
lams.png is the sub-figure (b) of Figure 13 in [3].


Step4: Dimensional Independence

	python meshinde.py

Results are generated in /NCP_MFVI/Darcy/RESULTS/Fig/NCPiMFVI.
*Readers can change the path by changing viariable `` RESULT_DIR, result_figs_dir''

change variable ``equ_nx''={40, 45, 50 ,55, 60}
load variables ``norm1, norm2, norm3, norm4, norm5''
load function ``plot_''
run ``plot_(0, 15)''

Figure:
inde.png is the sub-figure (c) of Figure 13 in [3].
