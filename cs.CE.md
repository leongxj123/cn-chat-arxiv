# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Era Splitting.](http://arxiv.org/abs/2309.14496) | 本研究提出了两种新的分裂准则，使得决策树模型能够利用时代信息进行优化，从而将超分布泛化研究中的思想应用于决策树模型。 |

# 详细

[^1]: Era Splitting.（arXiv:2309.14496v1 [cs.LG]）

    Era Splitting. (arXiv:2309.14496v1 [cs.LG])

    [http://arxiv.org/abs/2309.14496](http://arxiv.org/abs/2309.14496)

    本研究提出了两种新的分裂准则，使得决策树模型能够利用时代信息进行优化，从而将超分布泛化研究中的思想应用于决策树模型。

    

    现实生活中的机器学习问题在时间和空间上会呈现出数据的分布变化。这种行为超出了传统的经验风险最小化范式的范围，该范式假设数据在时间和地点上是独立同分布的。新兴的超分布泛化领域通过将环境或时代信息融入算法中，来应对这个现实。迄今为止，大部分研究都集中在线性模型和/或神经网络上。在本研究中，我们针对决策树模型，包括随机森林和梯度提升决策树，开发了两种新的分裂准则，使得树模型能够利用与每个数据点相关的时代信息，来找到在数据的所有不相交时代中都是最优的切分点，从而将超分布泛化研究中的思想应用于决策树模型。

    Real life machine learning problems exhibit distributional shifts in the data from one time to another or from on place to another. This behavior is beyond the scope of the traditional empirical risk minimization paradigm, which assumes i.i.d. distribution of data over time and across locations. The emerging field of out-of-distribution (OOD) generalization addresses this reality with new theory and algorithms which incorporate environmental, or era-wise information into the algorithms. So far, most research has been focused on linear models and/or neural networks. In this research we develop two new splitting criteria for decision trees, which allow us to apply ideas from OOD generalization research to decision tree models, including random forest and gradient-boosting decision trees. The new splitting criteria use era-wise information associated with each data point to allow tree-based models to find split points that are optimal across all disjoint eras in the data, instead of optim
    

