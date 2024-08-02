# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Application of Transformers for Nonlinear Channel Compensation in Optical Systems.](http://arxiv.org/abs/2304.13119) | 本文提出了一种利用Transformer进行光学系统非线性通道补偿的新方法，这种方法利用了Transformer的记忆关注能力和并行结构，实现了高效的非线性补偿。同时，作者还提出了一种物理学信息掩码，用于降低计算复杂度。 |

# 详细

[^1]: 利用Transformer进行光学系统非线性通道补偿的应用

    Application of Transformers for Nonlinear Channel Compensation in Optical Systems. (arXiv:2304.13119v1 [cs.IT])

    [http://arxiv.org/abs/2304.13119](http://arxiv.org/abs/2304.13119)

    本文提出了一种利用Transformer进行光学系统非线性通道补偿的新方法，这种方法利用了Transformer的记忆关注能力和并行结构，实现了高效的非线性补偿。同时，作者还提出了一种物理学信息掩码，用于降低计算复杂度。

    

    本文介绍了一种基于Transformer的新型非线性通道均衡方法，用于相干长距离传输。我们证明，由于其能够直接关注一系列符号之间的记忆，因此Transformer可以与并行结构有效地配合使用。我们展示了一个编码器部分的Transformer实现，用于非线性均衡，并分析了其在不同超参数范围内的性能。通过在每次迭代中处理符号块，并仔细选择要一起处理的编码器输出子集，可以实现高效的非线性补偿。我们还提出了一种基于非线性扰动理论的物理学信息掩码，用于降低Transformer非线性均衡的计算复杂度。

    In this paper, we introduce a new nonlinear channel equalization method for the coherent long-haul transmission based on Transformers. We show that due to their capability to attend directly to the memory across a sequence of symbols, Transformers can be used effectively with a parallelized structure. We present an implementation of encoder part of Transformer for nonlinear equalization and analyze its performance over a wide range of different hyper-parameters. It is shown that by processing blocks of symbols at each iteration and carefully selecting subsets of the encoder's output to be processed together, an efficient nonlinear compensation can be achieved. We also propose the use of a physic-informed mask inspired by nonlinear perturbation theory for reducing the computational complexity of Transformer nonlinear equalization.
    

