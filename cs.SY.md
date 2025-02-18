# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-Parametric Learning of Stochastic Differential Equations with Fast Rates of Convergence.](http://arxiv.org/abs/2305.15557) | 提出了一种新的非参数方法，用于识别随机微分方程中的漂移和扩散系数，该方法具有快速的收敛率，使得学习速率随着未知系数的光滑度增加而变得更加紧密。 |
| [^2] | [A Review on Longitudinal Car-Following Model.](http://arxiv.org/abs/2304.07143) | 这篇论文综述了逐车跟驰模型的不同原则和分类，以及面临的挑战和局限性。 |

# 详细

[^1]: 非参数学习具有快速收敛率的随机微分方程

    Non-Parametric Learning of Stochastic Differential Equations with Fast Rates of Convergence. (arXiv:2305.15557v1 [cs.LG])

    [http://arxiv.org/abs/2305.15557](http://arxiv.org/abs/2305.15557)

    提出了一种新的非参数方法，用于识别随机微分方程中的漂移和扩散系数，该方法具有快速的收敛率，使得学习速率随着未知系数的光滑度增加而变得更加紧密。

    

    我们提出了一种新颖的非参数学习范式来识别非线性随机微分方程的漂移和扩散系数，该范式依赖于状态的离散时间观测。其关键思想是将相应的Fokker-Planck方程的基于RKHS的近似拟合到这些观测值，从而得出理论学习速率的估计值，这与以往的工作不同，当未知漂移和扩散系数的光滑度越高时，理论估计值越来越紧。由于我们的方法是基于内核的，因此离线预处理可以在原则上得到有效的数值实现。

    We propose a novel non-parametric learning paradigm for the identification of drift and diffusion coefficients of non-linear stochastic differential equations, which relies upon discrete-time observations of the state. The key idea essentially consists of fitting a RKHS-based approximation of the corresponding Fokker-Planck equation to such observations, yielding theoretical estimates of learning rates which, unlike previous works, become increasingly tighter when the regularity of the unknown drift and diffusion coefficients becomes higher. Our method being kernel-based, offline pre-processing may in principle be profitably leveraged to enable efficient numerical implementation.
    
[^2]: 逐车跟驰模型综述

    A Review on Longitudinal Car-Following Model. (arXiv:2304.07143v1 [eess.SY])

    [http://arxiv.org/abs/2304.07143](http://arxiv.org/abs/2304.07143)

    这篇论文综述了逐车跟驰模型的不同原则和分类，以及面临的挑战和局限性。

    

    车跟车模型是交通仿真的核心组成部分，已经内置于许多配备ADAS的汽车中。对车跟车行为的研究使我们能够确定由基本的车辆交互过程引起的不同宏观现象的根源。本文提供了一份详尽的调查，重点介绍了各种车跟车模型之间的区别、互补性和重叠之处。该审查将在不同原则中概念化的车跟车模型进行分类。

    The car-following (CF) model is the core component for traffic simulations and has been built-in in many production vehicles with Advanced Driving Assistance Systems (ADAS). Research of CF behavior allows us to identify the sources of different macro phenomena induced by the basic process of pairwise vehicle interaction. The CF behavior and control model encompasses various fields, such as traffic engineering, physics, cognitive science, machine learning, and reinforcement learning. This paper provides a comprehensive survey highlighting differences, complementarities, and overlaps among various CF models according to their underlying logic and principles. We reviewed representative algorithms, ranging from the theory-based kinematic models, stimulus-response models, and cruise control models to data-driven Behavior Cloning (BC) and Imitation Learning (IL) and outlined their strengths and limitations. This review categorizes CF models that are conceptualized in varying principles and s
    

