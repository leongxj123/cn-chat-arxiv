# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic dimensionality reduction of Twin-in-the-Loop Observers.](http://arxiv.org/abs/2401.10945) | 本论文提出了一种自动降维的方法来解决车辆动力学估计中的各个变量独立计算和校准的问题，通过将经典控制取向车辆模型替换为车辆模拟器或数字双胞胎(DT)来实现，然后使用贝叶斯优化来调节滤波器。 |
| [^2] | [Maximum Causal Entropy Inverse Reinforcement Learning for Mean-Field Games.](http://arxiv.org/abs/2401.06566) | 本文介绍了最大因果熵逆强化学习（IRL）方法用于均场博弈（MFG）问题，提出了将MFG问题转化为广义纳什均衡问题（GNEP）的新算法。 |

# 详细

[^1]: Twin-in-the-Loop Observers的自动降维

    Automatic dimensionality reduction of Twin-in-the-Loop Observers. (arXiv:2401.10945v1 [cs.SY])

    [http://arxiv.org/abs/2401.10945](http://arxiv.org/abs/2401.10945)

    本论文提出了一种自动降维的方法来解决车辆动力学估计中的各个变量独立计算和校准的问题，通过将经典控制取向车辆模型替换为车辆模拟器或数字双胞胎(DT)来实现，然后使用贝叶斯优化来调节滤波器。

    

    目前车辆动力学估计技术通常存在一个共同的缺点：每个要估计的变量都是用独立的简化滤波模块计算的。这些模块并行运行并需要单独校准。为了解决这个问题，最近提出了一种统一的Twin-in-the-Loop(TiL)观测器架构：估计器中的经典简化控制取向车辆模型被一个完整的车辆模拟器或数字双胞胎(DT)替代。DT的状态通过线性时不变的输出误差定律实时校正。由于模拟器是一个黑盒子，没有明确的分析公式可用，因此无法使用经典的滤波器调节技术。出于这个原因，贝叶斯优化将用于解决一个数据驱动的优化问题来调节滤波器。由于DT的复杂性，优化问题是高维的。本文旨在找到一种调节高复杂度观测器的流程。

    State-of-the-art vehicle dynamics estimation techniques usually share one common drawback: each variable to estimate is computed with an independent, simplified filtering module. These modules run in parallel and need to be calibrated separately. To solve this issue, a unified Twin-in-the-Loop (TiL) Observer architecture has recently been proposed: the classical simplified control-oriented vehicle model in the estimators is replaced by a full-fledged vehicle simulator, or digital twin (DT). The states of the DT are corrected in real time with a linear time invariant output error law. Since the simulator is a black-box, no explicit analytical formulation is available, hence classical filter tuning techniques cannot be used. Due to this reason, Bayesian Optimization will be used to solve a data-driven optimization problem to tune the filter. Due to the complexity of the DT, the optimization problem is high-dimensional. This paper aims to find a procedure to tune the high-complexity obser
    
[^2]: 最大因果熵逆强化学习用于均场博弈问题

    Maximum Causal Entropy Inverse Reinforcement Learning for Mean-Field Games. (arXiv:2401.06566v1 [eess.SY])

    [http://arxiv.org/abs/2401.06566](http://arxiv.org/abs/2401.06566)

    本文介绍了最大因果熵逆强化学习（IRL）方法用于均场博弈（MFG）问题，提出了将MFG问题转化为广义纳什均衡问题（GNEP）的新算法。

    

    本文介绍了在无限时间间隔折扣回报最优性准则下，针对离散时间均场博弈（MFG）的最大因果熵逆强化学习（IRL）问题。典型智能体的状态空间是有限的。我们的方法首先全面回顾了关于确定性和随机马尔科夫决策过程（MDPs）在有限和无限时间间隔情

    In this paper, we introduce the maximum casual entropy Inverse Reinforcement Learning (IRL) problem for discrete-time mean-field games (MFGs) under an infinite-horizon discounted-reward optimality criterion. The state space of a typical agent is finite. Our approach begins with a comprehensive review of the maximum entropy IRL problem concerning deterministic and stochastic Markov decision processes (MDPs) in both finite and infinite-horizon scenarios. Subsequently, we formulate the maximum casual entropy IRL problem for MFGs - a non-convex optimization problem with respect to policies. Leveraging the linear programming formulation of MDPs, we restructure this IRL problem into a convex optimization problem and establish a gradient descent algorithm to compute the optimal solution with a rate of convergence. Finally, we present a new algorithm by formulating the MFG problem as a generalized Nash equilibrium problem (GNEP), which is capable of computing the mean-field equilibrium (MFE) f
    

