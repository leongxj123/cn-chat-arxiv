# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Risk-Sensitive Reinforcement Learning with Exponential Criteria.](http://arxiv.org/abs/2212.09010) | 本文介绍了一种风险敏感的强化学习算法，使用指数判据来提高其系统抗干扰性和实用性。作者进行了在模拟和实际机器人上的实验验证，表明该算法能够有效地提高样本效率和执行效果。 |

# 详细

[^1]: 风险敏感的强化学习算法：指数标准的应用

    Risk-Sensitive Reinforcement Learning with Exponential Criteria. (arXiv:2212.09010v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.09010](http://arxiv.org/abs/2212.09010)

    本文介绍了一种风险敏感的强化学习算法，使用指数判据来提高其系统抗干扰性和实用性。作者进行了在模拟和实际机器人上的实验验证，表明该算法能够有效地提高样本效率和执行效果。

    

    尽管风险中性的强化学习已经在很多应用中得到了实验成功，但是这种方法容易受到噪声和系统参数扰动的影响而不够稳健。因此,对风险敏感的强化学习算法进行了研究，以提高其系统抗干扰性，样本效率和实用性。本文介绍了一种新型的无模型风险敏感学习算法，将广泛使用的策略梯度算法进行变体，其实现过程类似。具体来说，本文研究了指数标准对强化学习代理的策略风险敏感性的影响，并开发了蒙特卡罗策略梯度算法和在线(时间差分)演员-评论家算法的变体。分析结果表明，指数标准的使用能够推广常用的特定正则化方法。作者在摆动杆和摆摆杆任务上进行了测试，验证了所提出的算法的实现性能和稳健性。

    While risk-neutral reinforcement learning has shown experimental success in a number of applications, it is well-known to be non-robust with respect to noise and perturbations in the parameters of the system. For this reason, risk-sensitive reinforcement learning algorithms have been studied to introduce robustness and sample efficiency, and lead to better real-life performance. In this work, we introduce new model-free risk-sensitive reinforcement learning algorithms as variations of widely-used Policy Gradient algorithms with similar implementation properties. In particular, we study the effect of exponential criteria on the risk-sensitivity of the policy of a reinforcement learning agent, and develop variants of the Monte Carlo Policy Gradient algorithm and the online (temporal-difference) Actor-Critic algorithm. Analytical results showcase that the use of exponential criteria generalize commonly used ad-hoc regularization approaches. The implementation, performance, and robustness 
    

