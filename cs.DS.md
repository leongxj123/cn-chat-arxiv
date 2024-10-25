# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Output-sensitive ERM-based techniques for data-driven algorithm design.](http://arxiv.org/abs/2204.03569) | 本研究通过列举问题实例总损失函数的部分来提出了基于输出感知ERM的数据驱动算法设计技术，解决了多参数组合算法族的计算效率问题。 |

# 详细

[^1]: 基于输出感知ERM的数据驱动算法设计技术

    Output-sensitive ERM-based techniques for data-driven algorithm design. (arXiv:2204.03569v2 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2204.03569](http://arxiv.org/abs/2204.03569)

    本研究通过列举问题实例总损失函数的部分来提出了基于输出感知ERM的数据驱动算法设计技术，解决了多参数组合算法族的计算效率问题。

    

    数据驱动算法设计是一种有潜力的基于学习的方法，用于超出最坏情况分析具有可调参数的算法。一个重要的开放问题是为具有多个参数的组合算法族设计计算效率高的数据驱动算法。当固定问题实例并变化参数时，"对偶"损失函数通常具有分段可分解的结构，即除了某些尖锐的转换边界外都表现良好。在本工作中，我们通过列举一组问题实例的总损失函数的部分来开展技术研究，以开发用于数据驱动算法设计的高效ERM学习算法。我们的方法的运行时间与实际出现的部分数目成比例，而不是基于部分数目的最坏情况上界。我们的方法涉及两个新颖的要素 - 一种用于枚举由一组超平面诱导的多面体的输出感知算法。

    Data-driven algorithm design is a promising, learning-based approach for beyond worst-case analysis of algorithms with tunable parameters. An important open problem is the design of computationally efficient data-driven algorithms for combinatorial algorithm families with multiple parameters. As one fixes the problem instance and varies the parameters, the "dual" loss function typically has a piecewise-decomposable structure, i.e. is well-behaved except at certain sharp transition boundaries. In this work we initiate the study of techniques to develop efficient ERM learning algorithms for data-driven algorithm design by enumerating the pieces of the sum dual loss functions for a collection of problem instances. The running time of our approach scales with the actual number of pieces that appear as opposed to worst case upper bounds on the number of pieces. Our approach involves two novel ingredients -- an output-sensitive algorithm for enumerating polytopes induced by a set of hyperpla
    

