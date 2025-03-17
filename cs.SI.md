# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nested stochastic block model for simultaneously clustering networks and nodes.](http://arxiv.org/abs/2307.09210) | 嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。 |
| [^2] | [Interdisciplinary Papers Supported by Disciplinary Grants Garner Deep and Broad Scientific Impact.](http://arxiv.org/abs/2303.14732) | 本研究使用测量框架，发现跨学科基金支持的跨学科论文与高影响力相关，但相较于学科基金，跨学科基金产生的支持论文更少且影响力下降。 |

# 详细

[^1]: 嵌套随机块模型用于同时对网络和节点进行聚类

    Nested stochastic block model for simultaneously clustering networks and nodes. (arXiv:2307.09210v1 [stat.ME])

    [http://arxiv.org/abs/2307.09210](http://arxiv.org/abs/2307.09210)

    嵌套随机块模型（NSBM）能够同时对网络和节点进行聚类，具有处理无标签网络、建模异质社群以及自动选择聚类数量的能力。

    

    我们引入了嵌套随机块模型（NSBM），用于对一组网络进行聚类，同时检测每个网络中的社群。NSBM具有几个吸引人的特点，包括能够处理具有潜在不同节点集的无标签网络，灵活地建模异质社群，以及自动选择网络类别和每个网络内社群数量的能力。通过贝叶斯模型实现这一目标，并将嵌套狄利克雷过程（NDP）作为先验，以联合建模网络间和网络内的聚类。网络数据引入的依赖性给NDP带来了非平凡的挑战，特别是在开发高效的采样器方面。对于后验推断，我们提出了几种马尔可夫链蒙特卡罗算法，包括标准的Gibbs采样器，简化Gibbs采样器和两种用于返回两个级别聚类结果的阻塞Gibbs采样器。

    We introduce the nested stochastic block model (NSBM) to cluster a collection of networks while simultaneously detecting communities within each network. NSBM has several appealing features including the ability to work on unlabeled networks with potentially different node sets, the flexibility to model heterogeneous communities, and the means to automatically select the number of classes for the networks and the number of communities within each network. This is accomplished via a Bayesian model, with a novel application of the nested Dirichlet process (NDP) as a prior to jointly model the between-network and within-network clusters. The dependency introduced by the network data creates nontrivial challenges for the NDP, especially in the development of efficient samplers. For posterior inference, we propose several Markov chain Monte Carlo algorithms including a standard Gibbs sampler, a collapsed Gibbs sampler, and two blocked Gibbs samplers that ultimately return two levels of clus
    
[^2]: 学科基金支持的跨学科论文具有深远的科学影响力

    Interdisciplinary Papers Supported by Disciplinary Grants Garner Deep and Broad Scientific Impact. (arXiv:2303.14732v1 [cs.DL])

    [http://arxiv.org/abs/2303.14732](http://arxiv.org/abs/2303.14732)

    本研究使用测量框架，发现跨学科基金支持的跨学科论文与高影响力相关，但相较于学科基金，跨学科基金产生的支持论文更少且影响力下降。

    

    跨学科研究已成为创新和发现的温床。基金支持的研究越来越占主导地位，与此同时，对资助跨学科工作的兴趣日益增长，这引发了关于跨学科拨款在支持高影响跨学科进展中的作用的根本问题。在这里，我们开发了一个测量框架，来量化研究基金和出版物的跨学科性，并将其应用于来自164个资助机构、26个国家的35万个拨款和从1985年到2009年承认这些拨款支持的130万篇论文。我们的分析揭示了两种矛盾的模式。一方面，跨学科基金往往会产生跨学科论文，而跨学科论文与高影响力相关。另一方面，与学科基金相比，跨学科基金产生的论文更少，而支持的跨学科论文影响力也大大下降。我们表明

    Interdisciplinary research has emerged as a hotbed for innovation and discoveries. The increasing dominance of grant-supported research, combined with growing interest in funding interdisciplinary work, raises fundamental questions on the role of interdisciplinary grants in supporting high-impact interdisciplinary advances. Here we develop a measurement framework to quantify the interdisciplinarity of both research grants and publications and apply it to 350K grants from 164 funding agencies over 26 countries and 1.3M papers that acknowledged the support of these grants from 1985 to 2009. Our analysis uncovers two contradictory patterns. On the one hand, interdisciplinary grants tend to produce interdisciplinary papers and interdisciplinary papers are associated with high impact. On the other hand, compared to their disciplinary counterparts, interdisciplinary grants produce much fewer papers and interdisciplinary papers that they support have substantially reduced impact. We show that
    

