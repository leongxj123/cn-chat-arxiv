# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Trained Random Forests Completely Reveal your Dataset](https://arxiv.org/abs/2402.19232) | 随机森林训练中没有采用自举聚合但具有特征随机化的模型容易被完全重建，即使采用自举聚合，大部分数据也可以被重建。 |
| [^2] | [Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective](https://arxiv.org/abs/2402.10686) | 通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险 |
| [^3] | [DPM: Clustering Sensitive Data through Separation.](http://arxiv.org/abs/2307.02969) | 本文提出了差分隐私聚类算法DPM，通过搜索准确的数据点分离器来进行隐私保护的聚类。关键贡献是识别大间隔分离器并合理分配隐私预算。 |

# 详细

[^1]: 训练的随机森林完全揭示您的数据集

    Trained Random Forests Completely Reveal your Dataset

    [https://arxiv.org/abs/2402.19232](https://arxiv.org/abs/2402.19232)

    随机森林训练中没有采用自举聚合但具有特征随机化的模型容易被完全重建，即使采用自举聚合，大部分数据也可以被重建。

    

    我们介绍了一种基于优化的重建攻击，能够完全或几乎完全重建用于训练随机森林的数据集。值得注意的是，我们的方法仅依赖于常用库（如scikit-learn）中随处可得的信息。为了实现这一点，我们将重建问题构建为一个组合问题，目标是最大似然。我们证明这个问题是NP难问题，但可以利用约束编程在规模上解决 —— 这是一种基于约束传播和解域缩减的方法。通过广泛的计算研究，我们证明没有采用自举聚合但具有特征随机化的随机森林容易被完全重建。即使使用少量树，这仍然成立。即使通过自举聚合，大部分数据也可以被重建。这些发现强调了一种关键。

    arXiv:2402.19232v1 Announce Type: new  Abstract: We introduce an optimization-based reconstruction attack capable of completely or near-completely reconstructing a dataset utilized for training a random forest. Notably, our approach relies solely on information readily available in commonly used libraries such as scikit-learn. To achieve this, we formulate the reconstruction problem as a combinatorial problem under a maximum likelihood objective. We demonstrate that this problem is NP-hard, though solvable at scale using constraint programming -- an approach rooted in constraint propagation and solution-domain reduction. Through an extensive computational investigation, we demonstrate that random forests trained without bootstrap aggregation but with feature randomization are susceptible to a complete reconstruction. This holds true even with a small number of trees. Even with bootstrap aggregation, the majority of the data can also be reconstructed. These findings underscore a critica
    
[^2]: 不确定性、校准和成员推理攻击：信息论视角

    Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective

    [https://arxiv.org/abs/2402.10686](https://arxiv.org/abs/2402.10686)

    通过信息论框架分析了最先进的似然比攻击对不确定性、校准水平和数据集大小的影响，研究了成员推理攻击中隐含的风险

    

    在成员推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否被用于训练目标模型。在本文中，我们在一个信息理论框架内分析了最先进的似然比攻击（LiRA）的性能，这个框架可以允许研究真实数据生成过程中的不确定性的影响，由有限训练数据集引起的认知不确定性以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收到的信息逐渐减少：置信向量（CV）披露，其中输出概率向量被发布；真实标签置信度（TLC）披露，其中只有模型分配给真实标签的概率是可用的；以及决策集（DS）披露。

    arXiv:2402.10686v1 Announce Type: cross  Abstract: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in 
    
[^3]: DPM: 通过分离聚类敏感数据

    DPM: Clustering Sensitive Data through Separation. (arXiv:2307.02969v1 [cs.CR])

    [http://arxiv.org/abs/2307.02969](http://arxiv.org/abs/2307.02969)

    本文提出了差分隐私聚类算法DPM，通过搜索准确的数据点分离器来进行隐私保护的聚类。关键贡献是识别大间隔分离器并合理分配隐私预算。

    

    隐私保护聚类以无监督方式对数据点进行分组，同时确保敏感信息得以保护。先前的隐私保护聚类关注点在于识别点云的聚集。本文则采取另一种方法，关注于识别适当的分离器以分离数据集。我们引入了新颖的差分隐私聚类算法DPM，以差分隐私的方式搜索准确的数据点分离器。DPM解决了寻找准确分离器的两个关键挑战：识别聚类间的大间隔分离器而不是聚类内的小间隔分离器，以及在开销隐私预算时，优先考虑将数据划分为较大子部分的分离器。利用差分隐私指数机制，DPM通过随机选择具有高效用性的聚类分离器：对于数据集D，如果中心的60%分位数中存在宽的低密度分离器，DPM会发现它。

    Privacy-preserving clustering groups data points in an unsupervised manner whilst ensuring that sensitive information remains protected. Previous privacy-preserving clustering focused on identifying concentration of point clouds. In this paper, we take another path and focus on identifying appropriate separators that split a data set. We introduce the novel differentially private clustering algorithm DPM that searches for accurate data point separators in a differentially private manner. DPM addresses two key challenges for finding accurate separators: identifying separators that are large gaps between clusters instead of small gaps within a cluster and, to efficiently spend the privacy budget, prioritising separators that split the data into large subparts. Using the differentially private Exponential Mechanism, DPM randomly chooses cluster separators with provably high utility: For a data set $D$, if there is a wide low-density separator in the central $60\%$ quantile, DPM finds that
    

