# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Communication Optimization for Distributed Training: Architecture, Advances, and Opportunities](https://arxiv.org/abs/2403.07585) | 本文介绍了分布式深度神经网络训练的通信优化架构，并对并行化策略、集体通信库和网络关系进行了分析，总结了当前的研究进展。 |

# 详细

[^1]: 分布式训练的通信优化：架构、进展和机遇

    Communication Optimization for Distributed Training: Architecture, Advances, and Opportunities

    [https://arxiv.org/abs/2403.07585](https://arxiv.org/abs/2403.07585)

    本文介绍了分布式深度神经网络训练的通信优化架构，并对并行化策略、集体通信库和网络关系进行了分析，总结了当前的研究进展。

    

    近年来，大规模深度神经网络模型的蓬勃发展，参数量不断增长。训练这些大规模模型通常需要庞大的内存和计算资源，超出了单个GPU的范围，需要进行分布式训练。由于近年来GPU性能迅速发展，计算时间缩短，因此通信在整体训练时间中的比例增加。因此，优化分布式训练的通信已经成为一个紧迫问题。本文简要介绍了分布式深度神经网络训练的总体架构，并从通信优化的角度分析了并行化策略、集体通信库和网络之间的关系，形成了一个三层范式。然后，我们回顾了当前具有代表性的研究进展与这个三层范式。我们发现lay

    arXiv:2403.07585v1 Announce Type: cross  Abstract: The past few years have witnessed the flourishing of large-scale deep neural network models with ever-growing parameter numbers. Training such large-scale models typically requires massive memory and computing resources that exceed those of a single GPU, necessitating distributed training. As GPU performance has rapidly evolved in recent years, computation time has shrunk, thereby increasing the proportion of communication in the overall training time. Therefore, optimizing communication for distributed training has become an urgent issue. In this article, we briefly introduce the general architecture of distributed deep neural network training and analyze relationships among Parallelization Strategy, Collective Communication Library, and Network from the perspective of communication optimization, which forms a three-layer paradigm. We then review current representative research advances with this three-layer paradigm. We find that lay
    

