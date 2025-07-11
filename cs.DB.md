# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network](https://arxiv.org/abs/2403.13268) | Unifews通过统一逐条稀疏化的方式，联合边权重稀疏化以提高学习效率，适用于不同架构设计并具有逐渐增加稀疏度的自适应压缩。 |
| [^2] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |

# 详细

[^1]: Unifews：用于高效图神经网络的统一逐条稀疏化

    Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network

    [https://arxiv.org/abs/2403.13268](https://arxiv.org/abs/2403.13268)

    Unifews通过统一逐条稀疏化的方式，联合边权重稀疏化以提高学习效率，适用于不同架构设计并具有逐渐增加稀疏度的自适应压缩。

    

    图神经网络（GNNs）在各种图学习任务中表现出了有希望的性能，但代价是资源密集型的计算。GNN更新的主要开销来自图传播和权重变换，两者都涉及对图规模矩阵的操作。先前的研究尝试通过利用图级别或网络级别的稀疏化技术来减少计算预算，从而产生缩小的图或权重。在这项工作中，我们提出了Unifews，它以逐个矩阵元素的方式统一了这两种操作，并进行联合边权重稀疏化以增强学习效率。Unifews的逐条设计使其能够在GNN层之间进行自适应压缩，稀疏度逐渐增加，并适用于各种架构设计，具有即时操作简化。在理论上，我们建立了一个新颖的框架来表征稀疏

    arXiv:2403.13268v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have shown promising performance in various graph learning tasks, but at the cost of resource-intensive computations. The primary overhead of GNN update stems from graph propagation and weight transformation, both involving operations on graph-scale matrices. Previous studies attempt to reduce the computational budget by leveraging graph-level or network-level sparsification techniques, resulting in downsized graph or weights. In this work, we propose Unifews, which unifies the two operations in an entry-wise manner considering individual matrix elements, and conducts joint edge-weight sparsification to enhance learning efficiency. The entry-wise design of Unifews enables adaptive compression across GNN layers with progressively increased sparsity, and is applicable to a variety of architectural designs with on-the-fly operation simplification. Theoretically, we establish a novel framework to characterize spa
    
[^2]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    

