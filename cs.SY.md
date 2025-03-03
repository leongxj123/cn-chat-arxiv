# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-Parametric Learning of Stochastic Differential Equations with Fast Rates of Convergence.](http://arxiv.org/abs/2305.15557) | 提出了一种新的非参数方法，用于识别随机微分方程中的漂移和扩散系数，该方法具有快速的收敛率，使得学习速率随着未知系数的光滑度增加而变得更加紧密。 |

# 详细

[^1]: 非参数学习具有快速收敛率的随机微分方程

    Non-Parametric Learning of Stochastic Differential Equations with Fast Rates of Convergence. (arXiv:2305.15557v1 [cs.LG])

    [http://arxiv.org/abs/2305.15557](http://arxiv.org/abs/2305.15557)

    提出了一种新的非参数方法，用于识别随机微分方程中的漂移和扩散系数，该方法具有快速的收敛率，使得学习速率随着未知系数的光滑度增加而变得更加紧密。

    

    我们提出了一种新颖的非参数学习范式来识别非线性随机微分方程的漂移和扩散系数，该范式依赖于状态的离散时间观测。其关键思想是将相应的Fokker-Planck方程的基于RKHS的近似拟合到这些观测值，从而得出理论学习速率的估计值，这与以往的工作不同，当未知漂移和扩散系数的光滑度越高时，理论估计值越来越紧。由于我们的方法是基于内核的，因此离线预处理可以在原则上得到有效的数值实现。

    We propose a novel non-parametric learning paradigm for the identification of drift and diffusion coefficients of non-linear stochastic differential equations, which relies upon discrete-time observations of the state. The key idea essentially consists of fitting a RKHS-based approximation of the corresponding Fokker-Planck equation to such observations, yielding theoretical estimates of learning rates which, unlike previous works, become increasingly tighter when the regularity of the unknown drift and diffusion coefficients becomes higher. Our method being kernel-based, offline pre-processing may in principle be profitably leveraged to enable efficient numerical implementation.
    

