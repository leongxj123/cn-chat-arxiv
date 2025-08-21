# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Energy-efficiency Limits on Training AI Systems using Learning-in-Memory](https://arxiv.org/abs/2402.14878) | 该论文提出了使用内存中学习的方法训练AI系统时的能效限制，并推导了新的理论下限。 |

# 详细

[^1]: 使用内存中学习的方法训练AI系统的能效限制

    Energy-efficiency Limits on Training AI Systems using Learning-in-Memory

    [https://arxiv.org/abs/2402.14878](https://arxiv.org/abs/2402.14878)

    该论文提出了使用内存中学习的方法训练AI系统时的能效限制，并推导了新的理论下限。

    

    arXiv:2402.14878v1 公告类型: cross 摘要: 内存中学习（LIM）是一种最近提出的范Paradigm，旨在克服训练机器学习系统中的基本内存瓶颈。虽然计算于内存（CIM）方法可以解决所谓的内存墙问题（即由于重复内存读取访问而消耗的能量），但它们对于以训练所需的精度重复内存写入时消耗的能量（更新墙）是不可知的，并且它们不考虑在短期和长期记忆之间传输信息时所消耗的能量（整合墙）。LIM范式提出，如果物理内存的能量屏障被自适应调制，使得存储器更新和整合的动态与梯度下降训练AI模型的Lyapunov动态相匹配，那么这些瓶颈也可以被克服。在本文中，我们推导了使用不同LIM应用程序训练AI系统时的能耗的新理论下限。

    arXiv:2402.14878v1 Announce Type: cross  Abstract: Learning-in-memory (LIM) is a recently proposed paradigm to overcome fundamental memory bottlenecks in training machine learning systems. While compute-in-memory (CIM) approaches can address the so-called memory-wall (i.e. energy dissipated due to repeated memory read access) they are agnostic to the energy dissipated due to repeated memory writes at the precision required for training (the update-wall), and they don't account for the energy dissipated when transferring information between short-term and long-term memories (the consolidation-wall). The LIM paradigm proposes that these bottlenecks, too, can be overcome if the energy barrier of physical memories is adaptively modulated such that the dynamics of memory updates and consolidation match the Lyapunov dynamics of gradient-descent training of an AI model. In this paper, we derive new theoretical lower bounds on energy dissipation when training AI systems using different LIM app
    

