# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nested stochastic block model for simultaneously clustering networks and nodes.](http://arxiv.org/abs/2307.09210) | 嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。 |

# 详细

[^1]: 嵌套随机块模型用于同时对网络和节点进行聚类

    Nested stochastic block model for simultaneously clustering networks and nodes. (arXiv:2307.09210v1 [stat.ME])

    [http://arxiv.org/abs/2307.09210](http://arxiv.org/abs/2307.09210)

    嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。

    

    我们引入了嵌套随机块模型（NSBM），用于对一组网络进行聚类，同时检测每个网络中的社群。NSBM具有几个吸引人的特点，包括能够处理具有潜在不同节点集的无标签网络，灵活地建模异质社群，以及自动选择网络类别和每个网络内社群数量的能力。通过贝叶斯模型实现这一目标，并将嵌套狄利克雷过程（NDP）作为先验，以联合建模网络间和网络内的聚类。网络数据引入的依赖性给NDP带来了非平凡的挑战，特别是在开发高效的采样器方面。对于后验推断，我们提出了几种马尔可夫链蒙特卡罗算法，包括标准的Gibbs采样器，简化Gibbs采样器和两种用于返回两个级别聚类结果的阻塞Gibbs采样器。

    We introduce the nested stochastic block model (NSBM) to cluster a collection of networks while simultaneously detecting communities within each network. NSBM has several appealing features including the ability to work on unlabeled networks with potentially different node sets, the flexibility to model heterogeneous communities, and the means to automatically select the number of classes for the networks and the number of communities within each network. This is accomplished via a Bayesian model, with a novel application of the nested Dirichlet process (NDP) as a prior to jointly model the between-network and within-network clusters. The dependency introduced by the network data creates nontrivial challenges for the NDP, especially in the development of efficient samplers. For posterior inference, we propose several Markov chain Monte Carlo algorithms including a standard Gibbs sampler, a collapsed Gibbs sampler, and two blocked Gibbs samplers that ultimately return two levels of clus
    

