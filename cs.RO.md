# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Elastic Time Steps](https://arxiv.org/abs/2402.14961) | SEAC是一种弹性时间步长的离策略演员-评论家算法，通过可变持续时间的时间步长，使代理能够根据情况改变控制频率，在模拟环境中表现优异。 |
| [^2] | [Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review](https://arxiv.org/abs/2402.10086) | 可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。 |

# 详细

[^1]: 弹性时间步长的强化学习

    Reinforcement Learning with Elastic Time Steps

    [https://arxiv.org/abs/2402.14961](https://arxiv.org/abs/2402.14961)

    SEAC是一种弹性时间步长的离策略演员-评论家算法，通过可变持续时间的时间步长，使代理能够根据情况改变控制频率，在模拟环境中表现优异。

    

    传统的强化学习（RL）算法通常应用于机器人学习以以固定控制频率执行动作的控制器。鉴于RL算法的离散性质，它们对控制频率的选择的影响视而不见：找到正确的控制频率可能很困难，错误往往会导致过度使用计算资源甚至导致无法收敛。我们提出了软弹性演员-评论家（SEAC）, 一种新颖的离策略演员-评论家算法来解决这个问题。SEAC实现了弹性时间步长，即具有已知变化持续时间的时间步长，允许代理根据情况改变其控制频率。在实践中，SEAC仅在必要时应用控制，最小化计算资源和数据使用。我们在模拟环境中评估了SEAC在牛顿运动学迷宫导航任务和三维赛车视频游戏Trackmania中的能力。SEAC在表现上优于SAC基线。

    arXiv:2402.14961v1 Announce Type: cross  Abstract: Traditional Reinforcement Learning (RL) algorithms are usually applied in robotics to learn controllers that act with a fixed control rate. Given the discrete nature of RL algorithms, they are oblivious to the effects of the choice of control rate: finding the correct control rate can be difficult and mistakes often result in excessive use of computing resources or even lack of convergence.   We propose Soft Elastic Actor-Critic (SEAC), a novel off-policy actor-critic algorithm to address this issue. SEAC implements elastic time steps, time steps with a known, variable duration, which allow the agent to change its control frequency to adapt to the situation. In practice, SEAC applies control only when necessary, minimizing computational resources and data usage.   We evaluate SEAC's capabilities in simulation in a Newtonian kinematics maze navigation task and on a 3D racing video game, Trackmania. SEAC outperforms the SAC baseline in t
    
[^2]: 可解释的人工智能在安全可信的自动驾驶中的应用：一项系统性评述

    Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review

    [https://arxiv.org/abs/2402.10086](https://arxiv.org/abs/2402.10086)

    可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。

    

    鉴于其在感知和规划任务中相对传统方法具有更优异的性能，人工智能（AI）对于自动驾驶（AD）的应用显示出了很大的潜力。然而，难以理解的AI系统加剧了对AD安全保证的现有挑战。缓解这一挑战的一种方法是利用可解释的AI（XAI）技术。为此，我们首次提出了关于可解释方法在安全可信的AD中的全面系统文献综述。我们首先分析了在AD背景下AI的要求，重点关注数据、模型和机构这三个关键方面。我们发现XAI对于满足这些要求是至关重要的。基于此，我们解释了AI中解释的来源，并描述了一种XAI的分类学。然后，我们确定了XAI在安全可信的AD中的五个主要贡献，包括可解释的设计、可解释的替代模型、可解释的监控，辅助...

    arXiv:2402.10086v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) shows promising applications for the perception and planning tasks in autonomous driving (AD) due to its superior performance compared to conventional methods. However, inscrutable AI systems exacerbate the existing challenge of safety assurance of AD. One way to mitigate this challenge is to utilize explainable AI (XAI) techniques. To this end, we present the first comprehensive systematic literature review of explainable methods for safe and trustworthy AD. We begin by analyzing the requirements for AI in the context of AD, focusing on three key aspects: data, model, and agency. We find that XAI is fundamental to meeting these requirements. Based on this, we explain the sources of explanations in AI and describe a taxonomy of XAI. We then identify five key contributions of XAI for safe and trustworthy AI in AD, which are interpretable design, interpretable surrogate models, interpretable monitoring, auxil
    

