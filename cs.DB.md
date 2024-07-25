# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [KIF: A Framework for Virtual Integration of Heterogeneous Knowledge Bases using Wikidata](https://arxiv.org/abs/2403.10304) | KIF框架利用Wikidata作为通用语言，结合用户定义的映射，实现了异构知识库的虚拟集成，形成类似于扩展Wikidata的虚拟知识库，可通过过滤接口或SPARQL进行查询。 |

# 详细

[^1]: KIF：使用Wikidata进行异构知识库虚拟集成的框架

    KIF: A Framework for Virtual Integration of Heterogeneous Knowledge Bases using Wikidata

    [https://arxiv.org/abs/2403.10304](https://arxiv.org/abs/2403.10304)

    KIF框架利用Wikidata作为通用语言，结合用户定义的映射，实现了异构知识库的虚拟集成，形成类似于扩展Wikidata的虚拟知识库，可通过过滤接口或SPARQL进行查询。

    

    我们提出了一个知识集成框架（称为KIF），该框架使用Wikidata作为通用语言来集成异构知识库。这些知识库可以是三元组存储、关系型数据库、CSV文件等，可以或不可以使用RDF的Wikidata方言。KIF利用Wikidata的数据模型和词汇以及用户定义的映射来展示集成库的统一视图，同时跟踪其陈述的上下文和出处。结果是一个行为类似于“扩展Wikidata”的虚拟知识库，可以通过高效过滤接口或使用SPARQL进行查询。我们展示了KIF的设计和实现，讨论了我们如何在化学领域（涉及Wikidata、PubChem和IBM CIRCA）中使用它解决实际集成问题，并介绍了KIF的性能和开销的实验结果。

    arXiv:2403.10304v1 Announce Type: new  Abstract: We present a knowledge integration framework (called KIF) that uses Wikidata as a lingua franca to integrate heterogeneous knowledge bases. These can be triplestores, relational databases, CSV files, etc., which may or may not use the Wikidata dialect of RDF. KIF leverages Wikidata's data model and vocabulary plus user-defined mappings to expose a unified view of the integrated bases while keeping track of the context and provenance of their statements. The result is a virtual knowledge base which behaves like an "extended Wikidata" and which can be queried either through an efficient filter interface or using SPARQL. We present the design and implementation of KIF, discuss how we have used it to solve a real integration problem in the domain of chemistry (involving Wikidata, PubChem, and IBM CIRCA), and present experimental results on the performance and overhead of KIF.
    

