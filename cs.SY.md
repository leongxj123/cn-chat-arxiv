# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Multi-modal Model Predictive Control via Duality-based Interaction Predictions](https://rss.arxiv.org/abs/2402.01116) | 我们提出了一个层级架构，通过使用对偶交互预测和精简的MPC问题，实现了可扩展的实时模型预测控制，在复杂的多模态交通场景中展示了12倍的速度提升。 |
| [^2] | [Learning Algorithms for Verification of Markov Decision Processes](https://arxiv.org/abs/2403.09184) | 该研究提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，旨在提高性能，避免对状态空间进行穷尽探索。 |

# 详细

[^1]: 可扩展多模型MPC的基于对偶交互预测的层级架构

    Scalable Multi-modal Model Predictive Control via Duality-based Interaction Predictions

    [https://rss.arxiv.org/abs/2402.01116](https://rss.arxiv.org/abs/2402.01116)

    我们提出了一个层级架构，通过使用对偶交互预测和精简的MPC问题，实现了可扩展的实时模型预测控制，在复杂的多模态交通场景中展示了12倍的速度提升。

    

    我们提出了一个层级架构，用于在复杂的多模态交通场景中实现可扩展的实时模型预测控制(MPC)。该架构由两个关键组件组成：1) RAID-Net，一种基于注意力机制的新颖循环神经网络，使用拉格朗日对偶性预测自动驾驶车辆与周围车辆之间在MPC预测范围内的相关交互；2) 一个简化的随机MPC问题，消除不相关的避碰约束，提高计算效率。我们的方法在一个模拟交通路口中演示，展示了解决运动规划问题的12倍速提升。您可以在这里找到展示该架构在多个复杂交通场景中的视频：https://youtu.be/-TcMeolCLWc

    We propose a hierarchical architecture designed for scalable real-time Model Predictive Control (MPC) in complex, multi-modal traffic scenarios. This architecture comprises two key components: 1) RAID-Net, a novel attention-based Recurrent Neural Network that predicts relevant interactions along the MPC prediction horizon between the autonomous vehicle and the surrounding vehicles using Lagrangian duality, and 2) a reduced Stochastic MPC problem that eliminates irrelevant collision avoidance constraints, enhancing computational efficiency. Our approach is demonstrated in a simulated traffic intersection with interactive surrounding vehicles, showcasing a 12x speed-up in solving the motion planning problem. A video demonstrating the proposed architecture in multiple complex traffic scenarios can be found here: https://youtu.be/-TcMeolCLWc
    
[^2]: 学习算法用于验证马尔可夫决策过程

    Learning Algorithms for Verification of Markov Decision Processes

    [https://arxiv.org/abs/2403.09184](https://arxiv.org/abs/2403.09184)

    该研究提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，旨在提高性能，避免对状态空间进行穷尽探索。

    

    我们提出了一个通用框架，将学习算法和启发式引导应用于马尔可夫决策过程（MDP）的验证，基于Br\'azdil, T.等人（2014）的想法。该框架的主要目标是通过避免对状态空间进行穷尽探索来提高性能，而是依靠启发式。本研究在很大程度上扩展了这种方法。对基础理论的几个细节进行了改进和错误修正。第1.3节提供了所有差异的概述。该框架专注于概率可达性，这是验证中的一个核心问题，并具体化为两种不同的场景。第一个假设完全了解MDP，尤其是精确的转移概率。它执行基于启发式的模型部分探索，产生精准的结果。

    arXiv:2403.09184v1 Announce Type: cross  Abstract: We present a general framework for applying learning algorithms and heuristical guidance to the verification of Markov decision processes (MDPs), based on the ideas of Br\'azdil, T. et al. (2014). Verification of Markov Decision Processes Using Learning Algorithms. The primary goal of the techniques presented in that work is to improve performance by avoiding an exhaustive exploration of the state space, guided by heuristics. This approach is significantly extended in this work. Several details of the base theory are refined and errors are fixed. Section 1.3 provides an overview of all differences.   The presented framework focuses on probabilistic reachability, which is a core problem in verification, and is instantiated in two distinct scenarios. The first assumes that full knowledge of the MDP is available, in particular precise transition probabilities. It performs a heuristic-driven partial exploration of the model, yielding preci
    

