# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers](https://arxiv.org/abs/2403.10266) | 动态序列并行性（DSP）为多维Transformer模型引入了一种高效的序列并行方法，通过动态切换并行维度实现对多维注意力模型的优化。 |

# 详细

[^1]: DSP：多维Transformer的动态序列并行性

    DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers

    [https://arxiv.org/abs/2403.10266](https://arxiv.org/abs/2403.10266)

    动态序列并行性（DSP）为多维Transformer模型引入了一种高效的序列并行方法，通过动态切换并行维度实现对多维注意力模型的优化。

    

    通过本文介绍的动态序列并行性（DSP）方法，可以为多维Transformer模型实现高效的序列并行性。其关键思想是根据当前计算阶段动态切换并行性维度，利用多维注意力的潜在特性。这种动态维度切换使得序列并行性在多维模型中具有最小的通信开销。

    arXiv:2403.10266v1 Announce Type: cross  Abstract: Scaling large models with long sequences across applications like language generation, video generation and multimodal tasks requires efficient sequence parallelism. However, existing sequence parallelism methods all assume a single sequence dimension and fail to adapt to multi-dimensional transformer architectures that perform attention calculations across different dimensions. This paper introduces Dynamic Sequence Parallelism (DSP), a novel approach to enable efficient sequence parallelism for multi-dimensional transformer models. The key idea is to dynamically switch the parallelism dimension according to the current computation stage, leveraging the potential characteristics of multi-dimensional attention. This dynamic dimension switching allows sequence parallelism with minimal communication overhead compared to applying traditional single-dimension parallelism to multi-dimensional models. Experiments show DSP improves end-to-end
    

