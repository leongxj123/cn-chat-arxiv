# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Fixed-Parameter Tractable Algorithm for Counting Markov Equivalence Classes with the same Skeleton.](http://arxiv.org/abs/2310.04218) | 本文提出了一个固定参数可处理算法，用于计数具有相同骨架的马尔可夫等价类。 |

# 详细

[^1]: 一个可计数具有相同骨架的马尔可夫等价类的固定参数可处理算法

    A Fixed-Parameter Tractable Algorithm for Counting Markov Equivalence Classes with the same Skeleton. (arXiv:2310.04218v1 [cs.DS])

    [http://arxiv.org/abs/2310.04218](http://arxiv.org/abs/2310.04218)

    本文提出了一个固定参数可处理算法，用于计数具有相同骨架的马尔可夫等价类。

    

    因果有向无环图（也称为贝叶斯网络）是编码随机变量之间条件依赖关系的流行工具。在因果有向无环图中，随机变量被建模为有向图中的顶点，并且规定每个随机变量在给定其父节点的情况下与其祖先节点无关。然而，对于同一组随机变量上的两个不同的因果有向无环图可以准确编码相同的一组条件依赖关系。这样的因果有向无环图被称为马尔可夫等价，马尔可夫等价的因果有向无环图的等价类被称为马尔可夫等价类（MEC）。在过去几十年中，对于MEC已经创建了一些美丽的组合特征，并且已知，特别是在同一MEC中的所有因果有向无环图必须具有相同的“骨架”（底层无向图）和v-结构（形式为$a\rightarrow b \leftarrow c$的诱导子图）。这些组合特征还提出了几个自然的算法问题。

    Causal DAGs (also known as Bayesian networks) are a popular tool for encoding conditional dependencies between random variables. In a causal DAG, the random variables are modeled as vertices in the DAG, and it is stipulated that every random variable is independent of its ancestors conditioned on its parents. It is possible, however, for two different causal DAGs on the same set of random variables to encode exactly the same set of conditional dependencies. Such causal DAGs are said to be Markov equivalent, and equivalence classes of Markov equivalent DAGs are known as Markov Equivalent Classes (MECs). Beautiful combinatorial characterizations of MECs have been developed in the past few decades, and it is known, in particular that all DAGs in the same MEC must have the same ''skeleton'' (underlying undirected graph) and v-structures (induced subgraph of the form $a\rightarrow b \leftarrow c$).  These combinatorial characterizations also suggest several natural algorithmic questions. On
    

