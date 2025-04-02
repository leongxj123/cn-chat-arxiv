# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Individualized Policy Evaluation and Learning under Clustered Network Interference](https://arxiv.org/abs/2311.02467) | 本文研究了集群网络干扰下个体化策略评估与学习的问题，提出了一种只假设半参数结构模型的方法，能够更准确地评估和学习最优的个体化处理规则。 |
| [^2] | [Discrete Hawkes process with flexible residual distribution and filtered historical simulation.](http://arxiv.org/abs/2401.13890) | 这篇论文介绍了一个灵活的离散Hawkes过程模型，可以集成不同的残差分布，并实现滤波历史模拟，扩展到多变量模型，与Hawkes过程相比，在高频金融数据估计上具有更好的效果。 |

# 详细

[^1]: 集群网络干扰下的个体化策略评估与学习

    Individualized Policy Evaluation and Learning under Clustered Network Interference

    [https://arxiv.org/abs/2311.02467](https://arxiv.org/abs/2311.02467)

    本文研究了集群网络干扰下个体化策略评估与学习的问题，提出了一种只假设半参数结构模型的方法，能够更准确地评估和学习最优的个体化处理规则。

    

    尽管现在有很多关于政策评估和学习的文献，但大部分之前的工作都假设一个个体的处理分配不会影响另一个个体的结果。不幸的是，忽视干扰可能导致评估偏误和无效的学习策略。例如，处理有很多朋友的有影响力的个体可能产生正向溢出效应，从而改善个体化处理规则（ITR）的整体性能。我们考虑在集群网络干扰（也称为部分干扰）下评估和学习最优ITR的问题，在该问题中，单位聚类从一个总体中抽样，并且在每个聚类中单位之间可能互相影响。与以前的方法强制限制溢出效应不同，所提出的方法只假设半参数结构模型，每个单位的结果是聚类中的个体处理的加法函数。

    While there now exists a large literature on policy evaluation and learning, much of prior work assumes that the treatment assignment of one unit does not affect the outcome of another unit. Unfortunately, ignoring interference may lead to biased policy evaluation and ineffective learned policies. For example, treating influential individuals who have many friends can generate positive spillover effects, thereby improving the overall performance of an individualized treatment rule (ITR). We consider the problem of evaluating and learning an optimal ITR under clustered network interference (also known as partial interference) where clusters of units are sampled from a population and units may influence one another within each cluster. Unlike previous methods that impose strong restrictions on spillover effects, the proposed methodology only assumes a semiparametric structural model where each unit's outcome is an additive function of individual treatments within the cluster. Under this 
    
[^2]: 灵活残差分布和滤波历史模拟的离散Hawkes过程

    Discrete Hawkes process with flexible residual distribution and filtered historical simulation. (arXiv:2401.13890v1 [q-fin.ST])

    [http://arxiv.org/abs/2401.13890](http://arxiv.org/abs/2401.13890)

    这篇论文介绍了一个灵活的离散Hawkes过程模型，可以集成不同的残差分布，并实现滤波历史模拟，扩展到多变量模型，与Hawkes过程相比，在高频金融数据估计上具有更好的效果。

    

    我们引入了一个新模型，可以将其视为离散意义上的Hawkes过程的扩展版本。该模型使得能够集成各种残差分布，同时保留原始Hawkes过程的基本属性。这种模型的丰富性使得滤波历史模拟能够更准确地融入原始时间序列的特性。该过程自然地扩展到容易实现估计和模拟的多变量模型。我们研究了灵活残差分布对高频金融数据估计的影响，与Hawkes过程进行了比较。

    We introduce a new model which can be considered as a extended version of the Hawkes process in a discrete sense. This model enables the integration of various residual distributions while preserving the fundamental properties of the original Hawkes process. The rich nature of this model enables a filtered historical simulation which incorporate the properties of original time series more accurately. The process naturally extends to multi-variate models with easy implementations of estimation and simulation. We investigate the effect of flexible residual distribution on estimation of high frequency financial data compared with the Hawkes process.
    

