# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Real-Time Recurrent Reinforcement Learning](https://arxiv.org/abs/2311.04830) | 本文提出了实时递归强化学习（RTRRL）方法，通过结合元-强化学习RNN架构、外部强化学习算法和RFLO局部在线学习，成功解决部分可观察马尔可夫决策过程中的离散和连续控制任务。实验结果表明，在计算复杂性相当的情况下，使用BPTT或RTRL替代RTRRL中的优化算法并不能提高回报。 |

# 详细

[^1]: 实时递归强化学习

    Real-Time Recurrent Reinforcement Learning

    [https://arxiv.org/abs/2311.04830](https://arxiv.org/abs/2311.04830)

    本文提出了实时递归强化学习（RTRRL）方法，通过结合元-强化学习RNN架构、外部强化学习算法和RFLO局部在线学习，成功解决部分可观察马尔可夫决策过程中的离散和连续控制任务。实验结果表明，在计算复杂性相当的情况下，使用BPTT或RTRL替代RTRRL中的优化算法并不能提高回报。

    

    在本文中，我们提出了实时递归强化学习（RTRRL），这是一种对部分可观察马尔可夫决策过程（POMDPs）中的离散和连续控制任务进行求解的生物学合理方法。RTRRL由三部分组成：（1）一个元-强化学习循环神经网络（RNN）架构，独立实现了一个演员-评论家算法；（2）一个外部强化学习算法，利用时序差分学习和荷兰资格追踪来训练元-强化学习网络；和（3）随机反馈局部在线（RFLO）学习，一种用于计算网络参数梯度的在线自动微分算法。我们的实验结果表明，通过将RTRRL中的优化算法替换为生物不合理的时延反向传播（BPTT）或实时递归学习（RTRL），并不能改善回报，同时在匹配BPTT的计算复杂性的情况下，甚至会增加返回。

    arXiv:2311.04830v2 Announce Type: replace  Abstract: In this paper we propose real-time recurrent reinforcement learning (RTRRL), a biologically plausible approach to solving discrete and continuous control tasks in partially-observable markov decision processes (POMDPs). RTRRL consists of three parts: (1) a Meta-RL RNN architecture, implementing on its own an actor-critic algorithm; (2) an outer reinforcement learning algorithm, exploiting temporal difference learning and dutch eligibility traces to train the Meta-RL network; and (3) random-feedback local-online (RFLO) learning, an online automatic differentiation algorithm for computing the gradients with respect to parameters of the network.Our experimental results show that by replacing the optimization algorithm in RTRRL with the biologically implausible back propagation through time (BPTT), or real-time recurrent learning (RTRL), one does not improve returns, while matching the computational complexity for BPTT, and even increasi
    

