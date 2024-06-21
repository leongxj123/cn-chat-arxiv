# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedStruct: Federated Decoupled Learning over Interconnected Graphs](https://arxiv.org/abs/2402.19163) | FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。 |
| [^2] | [On the Inductive Biases of Demographic Parity-based Fair Learning Algorithms](https://arxiv.org/abs/2402.18129) | 分析了标准 DP 基础正则化方法对给定敏感属性的预测标签条件分布的影响，并提出了一种基于敏感属性的分布稳健优化方法来控制归纳偏差。 |
| [^3] | [Top-$K$ ranking with a monotone adversary](https://arxiv.org/abs/2402.07445) | 本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。 |

# 详细

[^1]: FedStruct：联合解耦学习在互联图上

    FedStruct: Federated Decoupled Learning over Interconnected Graphs

    [https://arxiv.org/abs/2402.19163](https://arxiv.org/abs/2402.19163)

    FedStruct提出了一种新的框架，利用深层结构依赖关系在互联图上进行联合解耦学习，有效地维护隐私并捕捉节点间的依赖关系。

    

    我们解决了分布在多个客户端上的图结构数据上的联合学习挑战。具体来说，我们关注互联子图的普遍情况，其中不同客户端之间的相互连接起着关键作用。我们提出了针对这种情况的一种新颖框架，名为FedStruct，它利用深层结构依赖关系。为了维护隐私，与现有方法不同，FedStruct消除了在客户端之间共享或生成敏感节点特征或嵌入的必要性。相反，它利用显式全局图结构信息来捕捉节点间的依赖关系。我们通过在六个数据集上进行的实验结果验证了FedStruct的有效性，展示了在各种情况下（包括不同数据分区方法、不同标签可用性以及客户个数的）接近于集中式方法的性能。

    arXiv:2402.19163v1 Announce Type: new  Abstract: We address the challenge of federated learning on graph-structured data distributed across multiple clients. Specifically, we focus on the prevalent scenario of interconnected subgraphs, where inter-connections between different clients play a critical role. We present a novel framework for this scenario, named FedStruct, that harnesses deep structural dependencies. To uphold privacy, unlike existing methods, FedStruct eliminates the necessity of sharing or generating sensitive node features or embeddings among clients. Instead, it leverages explicit global graph structure information to capture inter-node dependencies. We validate the effectiveness of FedStruct through experimental results conducted on six datasets for semi-supervised node classification, showcasing performance close to the centralized approach across various scenarios, including different data partitioning methods, varying levels of label availability, and number of cl
    
[^2]: 关于基于人口统计学平等的公平学习算法的归纳偏差

    On the Inductive Biases of Demographic Parity-based Fair Learning Algorithms

    [https://arxiv.org/abs/2402.18129](https://arxiv.org/abs/2402.18129)

    分析了标准 DP 基础正则化方法对给定敏感属性的预测标签条件分布的影响，并提出了一种基于敏感属性的分布稳健优化方法来控制归纳偏差。

    

    公平的监督式学习算法在机器学习领域备受关注，这些算法在分配标签时很少依赖敏感属性。本文分析了标准DP（人口统计学平等）基础正则化方法对给定敏感属性的预测标签条件分布的影响。我们的分析表明，在具有非均匀分布敏感属性的训练数据集中，可能会导致分类规则偏向占据大多数训练数据的敏感属性结果。为了控制DP-based公平学习中的这种归纳偏差，我们提出了基于敏感属性的分布稳健优化（SA）

    arXiv:2402.18129v1 Announce Type: cross  Abstract: Fair supervised learning algorithms assigning labels with little dependence on a sensitive attribute have attracted great attention in the machine learning community. While the demographic parity (DP) notion has been frequently used to measure a model's fairness in training fair classifiers, several studies in the literature suggest potential impacts of enforcing DP in fair learning algorithms. In this work, we analytically study the effect of standard DP-based regularization methods on the conditional distribution of the predicted label given the sensitive attribute. Our analysis shows that an imbalanced training dataset with a non-uniform distribution of the sensitive attribute could lead to a classification rule biased toward the sensitive attribute outcome holding the majority of training data. To control such inductive biases in DP-based fair learning, we propose a sensitive attribute-based distributionally robust optimization (SA
    
[^3]: 具有单调对手的Top-K排名问题

    Top-$K$ ranking with a monotone adversary

    [https://arxiv.org/abs/2402.07445](https://arxiv.org/abs/2402.07445)

    本文针对具有单调对手的Top-K排名问题，提出了一种加权最大似然估计器(MLE)，在样本复杂度方面接近最优。算法创新包括了对加权MLE的精确且紧密的$\ell_\infty$误差分析，并与加权比较图的谱特性相关联。

    

    本文解决了具有单调对手的Top-K排名问题。我们考虑了一个比较图被随机生成且对手可以添加任意边的情况。统计学家的目标是根据从这个半随机比较图导出的两两比较准确地识别出Top-K的首选项。本文的主要贡献是开发出一种加权最大似然估计器(MLE)，它在样本复杂度方面达到了近似最优，最多差一个$log^2(n)$的因子，其中n表示比较项的数量。这得益于分析和算法创新的结合。在分析方面，我们提供了一种更明确、更紧密的加权MLE的$\ell_\infty$误差分析，它与加权比较图的谱特性相关。受此启发，我们的算法创新涉及到了

    In this paper, we address the top-$K$ ranking problem with a monotone adversary. We consider the scenario where a comparison graph is randomly generated and the adversary is allowed to add arbitrary edges. The statistician's goal is then to accurately identify the top-$K$ preferred items based on pairwise comparisons derived from this semi-random comparison graph. The main contribution of this paper is to develop a weighted maximum likelihood estimator (MLE) that achieves near-optimal sample complexity, up to a $\log^2(n)$ factor, where n denotes the number of items under comparison. This is made possible through a combination of analytical and algorithmic innovations. On the analytical front, we provide a refined $\ell_\infty$ error analysis of the weighted MLE that is more explicit and tighter than existing analyses. It relates the $\ell_\infty$ error with the spectral properties of the weighted comparison graph. Motivated by this, our algorithmic innovation involves the development 
    

