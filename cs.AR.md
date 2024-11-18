# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters](https://arxiv.org/abs/2402.10930) | ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。 |

# 详细

[^1]: ConSmax: 具有可学习参数的硬件友好型Softmax替代方案

    ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters

    [https://arxiv.org/abs/2402.10930](https://arxiv.org/abs/2402.10930)

    ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。

    

    自注意机制将基于transformer的大型语言模型（LLM）与卷积和循环神经网络区分开来。尽管性能有所提升，但由于自注意中广泛使用Softmax，在硅上实现实时LLM推断仍具挑战性。为了解决这一挑战，我们提出了Constant Softmax（ConSmax），这是一种高效的Softmax替代方案，采用可微的规范化参数来消除Softmax中的最大搜索和分母求和，实现了大规模并行化。

    arXiv:2402.10930v1 Announce Type: cross  Abstract: The self-attention mechanism sets transformer-based large language model (LLM) apart from the convolutional and recurrent neural networks. Despite the performance improvement, achieving real-time LLM inference on silicon is challenging due to the extensively used Softmax in self-attention. Apart from the non-linearity, the low arithmetic intensity greatly reduces the processing parallelism, which becomes the bottleneck especially when dealing with a longer context. To address this challenge, we propose Constant Softmax (ConSmax), a software-hardware co-design as an efficient Softmax alternative. ConSmax employs differentiable normalization parameters to remove the maximum searching and denominator summation in Softmax. It allows for massive parallelization while performing the critical tasks of Softmax. In addition, a scalable ConSmax hardware utilizing a bitwidth-split look-up table (LUT) can produce lossless non-linear operation and 
    

