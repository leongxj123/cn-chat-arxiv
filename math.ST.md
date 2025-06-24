# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^2] | [Indeterminate Probability Neural Network.](http://arxiv.org/abs/2303.11536) | 本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。 |
| [^3] | [Stable and consistent density-based clustering via multiparameter persistence.](http://arxiv.org/abs/2005.09048) | 这篇论文通过引入一种度量层次聚类的对应交错距离，研究了一种稳定一致的密度-based聚类算法，提供了一个从一参数层次聚类中提取单个聚类的算法，并证明了该算法的一致性和稳定性。 |

# 详细

[^1]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^2]: 不定概率神经网络

    Indeterminate Probability Neural Network. (arXiv:2303.11536v1 [cs.LG])

    [http://arxiv.org/abs/2303.11536](http://arxiv.org/abs/2303.11536)

    本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。

    

    本文提出了一个称为IPNN的新型通用模型，它将神经网络和概率论结合在一起。在传统概率论中，概率的计算是基于事件的发生，而这在当前的神经网络中几乎不使用。因此，我们提出了一种新的概率理论，它是经典概率论的扩展，并使经典概率论成为我们理论的一种特殊情况。此外，对于我们提出的神经网络框架，神经网络的输出被定义为概率事件，并基于这些事件的统计分析，推导出分类任务的推理模型。IPNN展现了新的特性：它在进行分类的同时可以执行无监督聚类。此外，IPNN能够使用非常小的神经网络进行非常大的分类，例如100个输出节点的模型可以分类10亿类别。理论优势体现在新的概率理论和神经网络框架中，并且实验结果展示了IPNN在各种应用中的潜力。

    We propose a new general model called IPNN - Indeterminate Probability Neural Network, which combines neural network and probability theory together. In the classical probability theory, the calculation of probability is based on the occurrence of events, which is hardly used in current neural networks. In this paper, we propose a new general probability theory, which is an extension of classical probability theory, and makes classical probability theory a special case to our theory. Besides, for our proposed neural network framework, the output of neural network is defined as probability events, and based on the statistical analysis of these events, the inference model for classification task is deduced. IPNN shows new property: It can perform unsupervised clustering while doing classification. Besides, IPNN is capable of making very large classification with very small neural network, e.g. model with 100 output nodes can classify 10 billion categories. Theoretical advantages are refl
    
[^3]: 稳定一致的密度-based聚类算法通过多参数持续性

    Stable and consistent density-based clustering via multiparameter persistence. (arXiv:2005.09048v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2005.09048](http://arxiv.org/abs/2005.09048)

    这篇论文通过引入一种度量层次聚类的对应交错距离，研究了一种稳定一致的密度-based聚类算法，提供了一个从一参数层次聚类中提取单个聚类的算法，并证明了该算法的一致性和稳定性。

    

    我们考虑了拓扑数据分析中的度-Rips构造，它提供了一种密度敏感的多参数层次聚类算法。我们使用我们引入的一种度量层次聚类的对应交错距离，分析了它对输入数据的扰动的稳定性。从度-Rips中取某些一参数切片可以恢复出已知的基于密度的聚类方法，但我们证明了这些方法是不稳定的。然而，我们证明了作为多参数对象的度-Rips是稳定的，并提出了一种从度-Rips中取切片的替代方法，该方法产生一个具有更好稳定性属性的一参数层次聚类算法。我们使用对应交错距离证明了该算法的一致性。我们提供了从一参数层次聚类中提取单个聚类的算法，该算法在对应交错距离方面是稳定的。

    We consider the degree-Rips construction from topological data analysis, which provides a density-sensitive, multiparameter hierarchical clustering algorithm. We analyze its stability to perturbations of the input data using the correspondence-interleaving distance, a metric for hierarchical clusterings that we introduce. Taking certain one-parameter slices of degree-Rips recovers well-known methods for density-based clustering, but we show that these methods are unstable. However, we prove that degree-Rips, as a multiparameter object, is stable, and we propose an alternative approach for taking slices of degree-Rips, which yields a one-parameter hierarchical clustering algorithm with better stability properties. We prove that this algorithm is consistent, using the correspondence-interleaving distance. We provide an algorithm for extracting a single clustering from one-parameter hierarchical clusterings, which is stable with respect to the correspondence-interleaving distance. And, we
    

