# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Serving Deep Learning Model in Relational Databases.](http://arxiv.org/abs/2310.04696) | 本文研究了在关系数据库中为深度学习模型提供服务的架构，并强调了三个关键范式：深度学习中心架构、UDF中心架构和关系中心架构。尽管每个架构都在特定的使用场景中有潜力，但还需要解决它们之间的集成问题和中间地带。 |

# 详细

[^1]: 在关系数据库中为深度学习模型提供服务

    Serving Deep Learning Model in Relational Databases. (arXiv:2310.04696v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2310.04696](http://arxiv.org/abs/2310.04696)

    本文研究了在关系数据库中为深度学习模型提供服务的架构，并强调了三个关键范式：深度学习中心架构、UDF中心架构和关系中心架构。尽管每个架构都在特定的使用场景中有潜力，但还需要解决它们之间的集成问题和中间地带。

    

    在不同商业和科学领域中，在关系数据上为深度学习模型提供服务已经成为一个重要需求，并引发了最近日益增长的兴趣。本文通过全面探索代表性架构来满足这个需求。我们强调了三个关键范式：尖端的深度学习中心架构将深度学习计算转移到专用的深度学习框架上。潜在的UDF中心架构将一个或多个张量计算封装到数据库系统中的用户定义函数(UDFs)中。潜在的关系中心架构旨在通过关系运算来表示大规模的张量计算。虽然每个架构在特定的使用场景中都显示出了潜力，但我们确定了这些架构之间的无缝集成和中间地带之间的紧急需求。我们深入研究了妨碍集成的差距，并探索了创新的策略。

    Serving deep learning (DL) models on relational data has become a critical requirement across diverse commercial and scientific domains, sparking growing interest recently. In this visionary paper, we embark on a comprehensive exploration of representative architectures to address the requirement. We highlight three pivotal paradigms: The state-of-the-artDL-Centricarchitecture offloadsDL computations to dedicated DL frameworks. The potential UDF-Centric architecture encapsulates one or more tensor computations into User Defined Functions (UDFs) within the database system. The potentialRelation-Centricarchitecture aims to represent a large-scale tensor computation through relational operators. While each of these architectures demonstrates promise in specific use scenarios, we identify urgent requirements for seamless integration of these architectures and the middle ground between these architectures. We delve into the gaps that impede the integration and explore innovative strategies 
    

