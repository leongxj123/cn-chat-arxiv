# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities](https://arxiv.org/abs/2403.07148) | 该论文针对三类变分不等式问题提出了具有随机重排的随机外推法（SEG-RR），并证明其在单调情况下实现了比均匀替换采样SEG更快的收敛速度。 |
| [^2] | [Optimal Queueing Regimes.](http://arxiv.org/abs/2401.13812) | 该论文研究了一个M/M/1排队模型，顾客可以根据策略决定加入队列或退出队列，并且给出了一类排队制度，使得在任何参数下，社会效率行为都是均衡结果。 |

# 详细

[^1]: 具有随机重排的随机外推法：改进变分不等式的收敛性

    Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities

    [https://arxiv.org/abs/2403.07148](https://arxiv.org/abs/2403.07148)

    该论文针对三类变分不等式问题提出了具有随机重排的随机外推法（SEG-RR），并证明其在单调情况下实现了比均匀替换采样SEG更快的收敛速度。

    

    随机外推法（SEG）方法是解决出现在各种机器学习任务中的有限求和极小-极大优化和变分不等式问题（VIPs）的最流行算法之一。然而，现有的SEG收敛分析专注于其带替换变体，而方法的实际实现会随机重新排列分量并按顺序使用它们。与广为研究的带替换变体不同，具有随机重排的SEG（SEG-RR）缺乏已建立的理论保证。在本工作中，我们针对三类VIPs（i）强单调，（ii）仿射和（iii）单调提供了SEG-RR的收敛性分析。我们推导了SEG-RR实现比均匀带替换采样SEG具有更快收敛速度的条件。在单调设置中，我们的SEG-RR分析保证了收敛到任意精度而无需大批量大小，这是对大批量大小而言的强要求。

    arXiv:2403.07148v1 Announce Type: cross  Abstract: The Stochastic Extragradient (SEG) method is one of the most popular algorithms for solving finite-sum min-max optimization and variational inequality problems (VIPs) appearing in various machine learning tasks. However, existing convergence analyses of SEG focus on its with-replacement variants, while practical implementations of the method randomly reshuffle components and sequentially use them. Unlike the well-studied with-replacement variants, SEG with Random Reshuffling (SEG-RR) lacks established theoretical guarantees. In this work, we provide a convergence analysis of SEG-RR for three classes of VIPs: (i) strongly monotone, (ii) affine, and (iii) monotone. We derive conditions under which SEG-RR achieves a faster convergence rate than the uniform with-replacement sampling SEG. In the monotone setting, our analysis of SEG-RR guarantees convergence to an arbitrary accuracy without large batch sizes, a strong requirement needed in 
    
[^2]: 最优排队制度

    Optimal Queueing Regimes. (arXiv:2401.13812v1 [econ.TH])

    [http://arxiv.org/abs/2401.13812](http://arxiv.org/abs/2401.13812)

    该论文研究了一个M/M/1排队模型，顾客可以根据策略决定加入队列或退出队列，并且给出了一类排队制度，使得在任何参数下，社会效率行为都是均衡结果。

    

    我们考虑了一个M/M/1排队模型，其中顾客可以根据策略决定是否加入队列、何时退出队列。我们对排队制度的类别进行了刻画，使得在模型的任何参数下，社会效率行为都是均衡结果。

    We consider an M/M/1 queueing model where customers can strategically decide whether to join the queue or balk and when to renege. We characterize the class of queueing regimes such that, for any parameters of the model, the socially efficient behavior is an equilibrium outcome.
    

