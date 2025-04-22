# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decidability of Querying First-Order Theories via Countermodels of Finite Width.](http://arxiv.org/abs/2304.06348) | 通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言 |
| [^2] | [Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points.](http://arxiv.org/abs/2303.12853) | 本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。 |

# 详细

[^1]: 通过有限宽度的反模型查询一阶理论的可决定性

    Decidability of Querying First-Order Theories via Countermodels of Finite Width. (arXiv:2304.06348v1 [cs.LO])

    [http://arxiv.org/abs/2304.06348](http://arxiv.org/abs/2304.06348)

    通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言

    

    我们提出了一个通用框架，基于具有结构简单的反模型的存在性（通过某些类型的宽度量来衡量，包括树宽和团宽等），为广泛的逻辑蕴含问题（简称查询）的可决定性提供了支持。作为我们框架的一个重要特例，我们确定了展现出宽度有限有限通用模型集的逻辑，保证了各种同态封闭查询的可决定性，包括了各种实际相关的查询语言。作为一个特别强大的宽度量，我们提出了Blumensath的分割宽度，该量包含了各种通常考虑的宽度量，具有非常有利的计算和结构特性。针对普遍展现存在性规则为一个展示案例，我们解释了有限分割宽度规则集包含其他已知的抽象可决定类，但借助现有的分层和受控规则集概念，也使我们能够捕获实际相关的查询语言，例如正则，连接和布尔连接查询。我们以存在规则的形式为重点，补充我们的理论结果，并进行了彻底的实验评估，展示了我们的框架在各种高级知识处理场景中的实际适用性和可伸缩性。

    We propose a generic framework for establishing the decidability of a wide range of logical entailment problems (briefly called querying), based on the existence of countermodels that are structurally simple, gauged by certain types of width measures (with treewidth and cliquewidth as popular examples). As an important special case of our framework, we identify logics exhibiting width-finite finitely universal model sets, warranting decidable entailment for a wide range of homomorphism-closed queries, subsuming a diverse set of practically relevant query languages. As a particularly powerful width measure, we propose Blumensath's partitionwidth, which subsumes various other commonly considered width measures and exhibits highly favorable computational and structural properties. Focusing on the formalism of existential rules as a popular showcase, we explain how finite partitionwidth sets of rules subsume other known abstract decidable classes but -- leveraging existing notions of strat
    
[^2]: 三次迭代的$(1-d)$-WL测试可以区分$d$维点云的非等距变换. (arXiv:2303.12853v1 [cs.LG])

    Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points. (arXiv:2303.12853v1 [cs.LG])

    [http://arxiv.org/abs/2303.12853](http://arxiv.org/abs/2303.12853)

    本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。

    

    Weisfeiler-Lehman (WL)测试是一个检查图同构的基本迭代算法。它被观察到是几种图神经网络体系结构设计的基础，这些网络的能力和性能可以用这个测试的表示能力来理解。受最近机器学习应用于涉及三维物体的数据集的发展启发，我们研究了当WL测试对完整的距离图表示的欧几里得点云是“完整的”时，它何时能够识别出任意一个任意点云.我们的主要结果是，$(d-1)$-维WL测试可以区分$d$维欧几里得空间中的点云，任何$d\ge 2$都可以，而且只需要进行三次测试。我们的结果对于$d=2,3$是紧的。我们还观察到$d$维WL测试只需要进行一次迭代就可以达到完整性。

    The Weisfeiler--Lehman (WL) test is a fundamental iterative algorithm for checking isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud.  Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and that only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.
    

