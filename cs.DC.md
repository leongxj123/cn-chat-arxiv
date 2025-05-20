# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CEFL: Carbon-Efficient Federated Learning.](http://arxiv.org/abs/2310.17972) | 该论文介绍了一种称为CEFL的碳高效联邦学习方法，通过使用自适应的成本感知策略来优化FL模型训练的任意成本度量，并成功实现了碳排放减少93％和训练时间减少50％的效果。 |

# 详细

[^1]: CEFL：碳高效的联邦学习

    CEFL: Carbon-Efficient Federated Learning. (arXiv:2310.17972v1 [cs.LG])

    [http://arxiv.org/abs/2310.17972](http://arxiv.org/abs/2310.17972)

    该论文介绍了一种称为CEFL的碳高效联邦学习方法，通过使用自适应的成本感知策略来优化FL模型训练的任意成本度量，并成功实现了碳排放减少93％和训练时间减少50％的效果。

    

    联邦学习（FL）通过将机器学习（ML）训练分布在许多边缘设备上，以减少数据传输开销和保护数据隐私。由于FL模型训练可能涉及数百万个设备，因此需要大量资源，因此之前的工作一直致力于提高其资源效率以优化时间至准确性。然而，之前的工作通常将所有资源视为相同，而实际上它们可能产生大不相同的成本，这反而激发了优化成本至准确性的动机。为了解决这个问题，我们设计了CEFL，它使用自适应的成本感知客户选择策略，在训练FL模型时优化任意成本度量。我们的策略扩展并结合了基于效用的客户选择和关键学习期的先前工作，使其具有成本感知性。我们通过设计碳高效的FL来演示CEFL，在这里能源的碳强度是成本，并且显示它可以将碳排放减少93％，并将训练时间减少50％，与随机客户相比。

    Federated Learning (FL) distributes machine learning (ML) training across many edge devices to reduce data transfer overhead and protect data privacy. Since FL model training may span millions of devices and is thus resource-intensive, prior work has focused on improving its resource efficiency to optimize time-to-accuracy. However, prior work generally treats all resources the same, while, in practice, they may incur widely different costs, which instead motivates optimizing cost-to-accuracy. To address the problem, we design CEFL, which uses adaptive cost-aware client selection policies to optimize an arbitrary cost metric when training FL models. Our policies extend and combine prior work on utility-based client selection and critical learning periods by making them cost-aware. We demonstrate CEFL by designing carbon-efficient FL, where energy's carbon-intensity is the cost, and show that it i) reduces carbon emissions by 93\% and reduces training time by 50% compared to random clie
    

