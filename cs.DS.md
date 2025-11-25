# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reserve Matching with Thresholds.](http://arxiv.org/abs/2309.13766) | 本文提出了一个新的智能保留系统，使用阈值模型允许独立的优先级排序和任意的受益人和资格阈值，以实现优化分配单元数的分配。 |
| [^2] | [Fairness in Streaming Submodular Maximization over a Matroid Constraint.](http://arxiv.org/abs/2305.15118) | 这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。 |

# 详细

[^1]: 阈值保留匹配

    Reserve Matching with Thresholds. (arXiv:2309.13766v1 [econ.TH])

    [http://arxiv.org/abs/2309.13766](http://arxiv.org/abs/2309.13766)

    本文提出了一个新的智能保留系统，使用阈值模型允许独立的优先级排序和任意的受益人和资格阈值，以实现优化分配单元数的分配。

    

    保留系统通过创建优先考虑各自受益人的类别，以适应多个重要或代表性不足的群体来分配不可分割的稀缺资源。一些应用包括疫苗的最优分配，或将少数人分配到印度的精英学院。智能分配是指优化分配单元数的分配。先前的文献大多假定了基准优先级，这在不同类别的优先级排序之间会产生重要的相互依赖。它还假定要么每个人都有资格从任何类别接收单元，要么只有受益人有资格。我们提出的全面阈值模型允许在类别之间进行独立的优先级排序，并设置任意的受益人和资格阈值，使决策者能够避免在平权行动系统中进行无法比较的比较。我们提出了一个新的智能保留系统，可以同时优化两个目标。

    Reserve systems are used to accommodate multiple essential or underrepresented groups in allocating indivisible scarce resources by creating categories that prioritize their respective beneficiaries. Some applications include the optimal allocation of vaccines, or assignment of minority students to elite colleges in India. An allocation is called smart if it optimizes the number of units distributed. Previous literature mostly assumed baseline priorities, which impose significant interdependencies between the priority ordering of different categories. It also assumes either everybody is eligible for receiving a unit from any category, or only the beneficiaries are eligible. The comprehensive Threshold Model we propose allows independent priority orderings among categories and arbitrary beneficiary and eligibility thresholds, enabling policymakers to avoid comparing incomparables in affirmative action systems. We present a new smart reserve system that optimizes two objectives simultane
    
[^2]: 在一个Matroid约束下流式子模最大化中的公平性

    Fairness in Streaming Submodular Maximization over a Matroid Constraint. (arXiv:2305.15118v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15118](http://arxiv.org/abs/2305.15118)

    这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。

    

    流式子模最大化是从一个大规模数据集中选择一个代表性子集的自然模型。如果数据点具有敏感属性，如性别或种族，强制公平性以避免偏见和歧视变得重要。这引起了对开发公平机器学习算法的极大兴趣。最近，这样的算法已经被开发用于基于基数约束的单调子模最大化。在本文中，我们研究了这个问题的自然推广到一个Matroid约束。我们提供了流式算法以及不可能的结果，这些结果在效率、质量和公平性之间提供了权衡。我们在一系列知名的现实世界应用中对我们的发现进行了经验证实：基于示例的聚类、电影推荐和社交网络中的最大覆盖。

    Streaming submodular maximization is a natural model for the task of selecting a representative subset from a large-scale dataset. If datapoints have sensitive attributes such as gender or race, it becomes important to enforce fairness to avoid bias and discrimination. This has spurred significant interest in developing fair machine learning algorithms. Recently, such algorithms have been developed for monotone submodular maximization under a cardinality constraint.  In this paper, we study the natural generalization of this problem to a matroid constraint. We give streaming algorithms as well as impossibility results that provide trade-offs between efficiency, quality and fairness. We validate our findings empirically on a range of well-known real-world applications: exemplar-based clustering, movie recommendation, and maximum coverage in social networks.
    

