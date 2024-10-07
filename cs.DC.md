# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scattered Mixture-of-Experts Implementation](https://arxiv.org/abs/2403.08245) | ScatterMoE是一种在GPU上实现的稀疏专家混合模型，通过避免填充和过多复制输入，提高了推理和训练速度，并减少了内存占用。 |

# 详细

[^1]: 分散式专家混合模型的实现

    Scattered Mixture-of-Experts Implementation

    [https://arxiv.org/abs/2403.08245](https://arxiv.org/abs/2403.08245)

    ScatterMoE是一种在GPU上实现的稀疏专家混合模型，通过避免填充和过多复制输入，提高了推理和训练速度，并减少了内存占用。

    

    我们提出了ScatterMoE，这是一种在GPU上实现的稀疏专家混合模型（SMoE）。ScatterMoE在现有实现的基础上构建，克服了一些限制以提高推理和训练速度，并减少内存占用。该实现通过避免填充和过多复制输入来实现这一目标。我们介绍了ParallelLinear，这是我们用来构建实现的主要组件，以及用于加速操作的各种内核。我们对我们的实现进行了与Megablocks的基准测试，并展示它可以实现更高的吞吐量和更低的内存占用。我们还展示了ParallelLinear如何通过展示Mixture of Attention的实现来扩展专家混合模型的概念。

    arXiv:2403.08245v1 Announce Type: new  Abstract: We present ScatterMoE, an implementation of Sparse Mixture-of-Experts (SMoE) on GPUs. ScatterMoE builds upon existing implementations, and overcoming some of the limitations to improve inference and training speed, and memory footprint. This implementation achieves this by avoiding padding and making excessive copies of the input.   We introduce ParallelLinear, the main component we use to build our implementation and the various kernels used to speed up the operation. We benchmark our implementation against Megablocks, and show that it enables a higher throughput and lower memory footprint. We also show how ParallelLinear enables extension of the Mixture-of-Experts concept by demonstrating with an implementation of Mixture of Attention.
    

