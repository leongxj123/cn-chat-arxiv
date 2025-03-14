# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Real-Time Recurrent Reinforcement Learning](https://arxiv.org/abs/2311.04830) | 本文提出了实时递归强化学习（RTRRL）方法，通过结合元-强化学习RNN架构、外部强化学习算法和RFLO局部在线学习，成功解决部分可观察马尔可夫决策过程中的离散和连续控制任务。实验结果表明，在计算复杂性相当的情况下，使用BPTT或RTRL替代RTRRL中的优化算法并不能提高回报。 |
| [^2] | [Networked Communication for Decentralised Agents in Mean-Field Games.](http://arxiv.org/abs/2306.02766) | 本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。 |
| [^3] | [Feasible Policy Iteration.](http://arxiv.org/abs/2304.08845) | 可行性策略迭代 (FPI) 是一个间接的安全强化学习方法，使用上一个策略的可行域来迭代地限制当前策略。可行性策略改进是其核心，它在可行域内最大化回报，在可行域外最小化约束衰减函数 (CDF). |

# 详细

[^1]: 实时递归强化学习

    Real-Time Recurrent Reinforcement Learning

    [https://arxiv.org/abs/2311.04830](https://arxiv.org/abs/2311.04830)

    本文提出了实时递归强化学习（RTRRL）方法，通过结合元-强化学习RNN架构、外部强化学习算法和RFLO局部在线学习，成功解决部分可观察马尔可夫决策过程中的离散和连续控制任务。实验结果表明，在计算复杂性相当的情况下，使用BPTT或RTRL替代RTRRL中的优化算法并不能提高回报。

    

    在本文中，我们提出了实时递归强化学习（RTRRL），这是一种对部分可观察马尔可夫决策过程（POMDPs）中的离散和连续控制任务进行求解的生物学合理方法。RTRRL由三部分组成：（1）一个元-强化学习循环神经网络（RNN）架构，独立实现了一个演员-评论家算法；（2）一个外部强化学习算法，利用时序差分学习和荷兰资格追踪来训练元-强化学习网络；和（3）随机反馈局部在线（RFLO）学习，一种用于计算网络参数梯度的在线自动微分算法。我们的实验结果表明，通过将RTRRL中的优化算法替换为生物不合理的时延反向传播（BPTT）或实时递归学习（RTRL），并不能改善回报，同时在匹配BPTT的计算复杂性的情况下，甚至会增加返回。

    arXiv:2311.04830v2 Announce Type: replace  Abstract: In this paper we propose real-time recurrent reinforcement learning (RTRRL), a biologically plausible approach to solving discrete and continuous control tasks in partially-observable markov decision processes (POMDPs). RTRRL consists of three parts: (1) a Meta-RL RNN architecture, implementing on its own an actor-critic algorithm; (2) an outer reinforcement learning algorithm, exploiting temporal difference learning and dutch eligibility traces to train the Meta-RL network; and (3) random-feedback local-online (RFLO) learning, an online automatic differentiation algorithm for computing the gradients with respect to parameters of the network.Our experimental results show that by replacing the optimization algorithm in RTRRL with the biologically implausible back propagation through time (BPTT), or real-time recurrent learning (RTRL), one does not improve returns, while matching the computational complexity for BPTT, and even increasi
    
[^2]: 分布式智能体在均场博弈中的网络通信

    Networked Communication for Decentralised Agents in Mean-Field Games. (arXiv:2306.02766v2 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2306.02766](http://arxiv.org/abs/2306.02766)

    本研究在均场博弈中引入网络通信，提出了一种提高分布式智能体学习效率的方案，并进行了实际实验验证。

    

    我们将网络通信引入均场博弈框架，特别是在无oracle的情况下，N个分布式智能体沿着经过的经验系统的单一非周期演化路径学习。我们证明，我们的架构在只有一些关于网络结构的合理假设的情况下，具有样本保证，在集中学习和独立学习情况之间有界。我们讨论了三个理论算法的样本保证实际上并不会导致实际收敛。因此，我们展示了在实际设置中，当理论参数未被观察到（导致Q函数的估计不准确）时，我们的通信方案显著加速了收敛速度，而无需依赖于一个不可取的集中式控制器的假设。我们对三个理论算法进行了几种实际的改进，使我们能够展示它们的第一个实证表现。

    We introduce networked communication to the mean-field game framework, in particular to oracle-free settings where $N$ decentralised agents learn along a single, non-episodic evolution path of the empirical system. We prove that our architecture, with only a few reasonable assumptions about network structure, has sample guarantees bounded between those of the centralised- and independent-learning cases. We discuss how the sample guarantees of the three theoretical algorithms do not actually result in practical convergence. Accordingly, we show that in practical settings where the theoretical parameters are not observed (leading to poor estimation of the Q-function), our communication scheme significantly accelerates convergence over the independent case, without relying on the undesirable assumption of a centralised controller. We contribute several further practical enhancements to all three theoretical algorithms, allowing us to showcase their first empirical demonstrations. Our expe
    
[^3]: 可行性策略迭代

    Feasible Policy Iteration. (arXiv:2304.08845v1 [cs.LG])

    [http://arxiv.org/abs/2304.08845](http://arxiv.org/abs/2304.08845)

    可行性策略迭代 (FPI) 是一个间接的安全强化学习方法，使用上一个策略的可行域来迭代地限制当前策略。可行性策略改进是其核心，它在可行域内最大化回报，在可行域外最小化约束衰减函数 (CDF).

    

    安全强化学习旨在在安全约束下解决最优控制问题。现有的 $\textit{直接}$ 安全强化学习方法会在整个学习过程中一直使用原始约束。它们或者缺乏策略迭代期间的理论保证，或者遭遇不可行性问题。为了解决这个问题，我们提出了一个叫做可行性策略迭代（FPI）的 $\textit{间接}$ 安全强化学习方法，它使用最后一个策略的可行域来迭代地限制当前策略。可行域由一个叫做约束衰减函数（CDF）的可行性函数表示。FPI 的核心是一个叫做可行性策略改进的区域性策略更新规则，它在可行域内最大化回报，在可行域外最小化 CDF。这个更新规则总是可行的，并确保可行域单调地扩展，状态值函数单调地增长。

    Safe reinforcement learning (RL) aims to solve an optimal control problem under safety constraints. Existing $\textit{direct}$ safe RL methods use the original constraint throughout the learning process. They either lack theoretical guarantees of the policy during iteration or suffer from infeasibility problems. To address this issue, we propose an $\textit{indirect}$ safe RL method called feasible policy iteration (FPI) that iteratively uses the feasible region of the last policy to constrain the current policy. The feasible region is represented by a feasibility function called constraint decay function (CDF). The core of FPI is a region-wise policy update rule called feasible policy improvement, which maximizes the return under the constraint of the CDF inside the feasible region and minimizes the CDF outside the feasible region. This update rule is always feasible and ensures that the feasible region monotonically expands and the state-value function monotonically increases inside 
    

