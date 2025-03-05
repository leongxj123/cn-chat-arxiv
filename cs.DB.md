# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Relational Patterns for Logical Query Answering over Knowledge Graphs.](http://arxiv.org/abs/2303.11858) | 本文介绍了一种新的查询嵌入方法RoConE，它允许学习关系模式并提高了逻辑查询推理的性能。 |

# 详细

[^1]: 对知识图谱进行逻辑查询应答的关系模式建模

    Modeling Relational Patterns for Logical Query Answering over Knowledge Graphs. (arXiv:2303.11858v1 [cs.DB])

    [http://arxiv.org/abs/2303.11858](http://arxiv.org/abs/2303.11858)

    本文介绍了一种新的查询嵌入方法RoConE，它允许学习关系模式并提高了逻辑查询推理的性能。

    

    对知识图谱（KG）进行一阶逻辑（FOL）查询的回答仍然是一项具有挑战性的任务，主要是由于KG不完整性而导致的。查询嵌入方法通过计算实体、关系和逻辑查询的低维度向量表示来解决这个问题。KG表现出对称性和组合性等关系模式，建模这些模式可以进一步提高查询嵌入模型的性能。然而，这些模式在查询嵌入模型中的作用尚未在文献中进行研究。在本文中，我们填补了这一研究空白，并通过引入允许学习关系模式的归纳偏差，加强FOL查询推理的模式推理能力。为此，我们开发了一种新的查询嵌入方法RoConE，它将查询区域定义为几何锥体，并通过在复杂空间中旋转代数查询算子。RoConE结合了几何锥体作为可以明确定义查询表示的几何表示和旋转代数查询算子的优势。

    Answering first-order logical (FOL) queries over knowledge graphs (KG) remains a challenging task mainly due to KG incompleteness. Query embedding approaches this problem by computing the low-dimensional vector representations of entities, relations, and logical queries. KGs exhibit relational patterns such as symmetry and composition and modeling the patterns can further enhance the performance of query embedding models. However, the role of such patterns in answering FOL queries by query embedding models has not been yet studied in the literature. In this paper, we fill in this research gap and empower FOL queries reasoning with pattern inference by introducing an inductive bias that allows for learning relation patterns. To this end, we develop a novel query embedding method, RoConE, that defines query regions as geometric cones and algebraic query operators by rotations in complex space. RoConE combines the advantages of Cone as a well-specified geometric representation for query e
    

