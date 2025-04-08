# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying the Cost of Learning in Queueing Systems.](http://arxiv.org/abs/2308.07817) | 本文提出了一种新的度量方法，即学习队列中的成本 (CLQ)，用于量化由参数不确定性导致的时间平均队列长度最大增加量。该度量方法可以捕捉学习队列系统的统计复杂性，不局限于渐近性能。 |

# 详细

[^1]: 量化队列系统中的学习成本

    Quantifying the Cost of Learning in Queueing Systems. (arXiv:2308.07817v1 [cs.LG])

    [http://arxiv.org/abs/2308.07817](http://arxiv.org/abs/2308.07817)

    本文提出了一种新的度量方法，即学习队列中的成本 (CLQ)，用于量化由参数不确定性导致的时间平均队列长度最大增加量。该度量方法可以捕捉学习队列系统的统计复杂性，不局限于渐近性能。

    

    队列系统是广泛应用的随机模型，应用于通信网络、医疗保健、服务系统等等。虽然它们的最优控制已经得到了广泛研究，但大多数现有方法都假设系统参数的完美知识。然而，在实践中，参数不确定性很常见，因此最近一系列关于队列系统的学习的研究产生了。这个新兴的研究方向主要关注所提算法的渐近性能。本文中，我们认为渐近度量，即着眼于后期性能的度量，无法捕捉学习队列系统中固有的统计复杂性，这种复杂性通常出现在早期阶段。相反，我们提出了学习队列中的成本 (CLQ)，这是一种新的度量方法，可以衡量由参数不确定性导致的时间平均队列长度的最大增加量。我们对单队列多服务器系统的CLQ进行了表征。

    Queueing systems are widely applicable stochastic models with use cases in communication networks, healthcare, service systems, etc. Although their optimal control has been extensively studied, most existing approaches assume perfect knowledge of system parameters. Of course, this assumption rarely holds in practice where there is parameter uncertainty, thus motivating a recent line of work on bandit learning for queueing systems. This nascent stream of research focuses on the asymptotic performance of the proposed algorithms.  In this paper, we argue that an asymptotic metric, which focuses on late-stage performance, is insufficient to capture the intrinsic statistical complexity of learning in queueing systems which typically occurs in the early stage. Instead, we propose the Cost of Learning in Queueing (CLQ), a new metric that quantifies the maximum increase in time-averaged queue length caused by parameter uncertainty. We characterize the CLQ of a single-queue multi-server system,
    

