# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compressing Transformer-based self-supervised models for speech processing.](http://arxiv.org/abs/2211.09949) | 本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。 |

# 详细

[^1]: 对基于Transformer的自监督模型在语音处理中进行压缩

    Compressing Transformer-based self-supervised models for speech processing. (arXiv:2211.09949v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09949](http://arxiv.org/abs/2211.09949)

    本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。

    

    尽管Transformer在自监督学习中取得了成功，并应用于各种下游任务，但是训练和推断的计算成本仍然是将这些模型应用于各种设备的主要挑战。目前已有一些孤立的尝试来压缩Transformer，但研究中的设置和指标各不相同。此前的工作很少涉及不同压缩率之间的权衡，这使得比较压缩技术变得困难。在这项工作中，我们旨在为这些孤立结果提供背景，研究几种常用的压缩技术，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。我们报告了在不同压缩率下的权衡，包括墙钟时间、参数数量和乘加操作数量。我们的结果表明，与最近的方法相比，基本的压缩技术是强大的基准。我们进一步提出了几种压缩方法来改进模型的压缩效果。

    Despite the success of Transformers in self- supervised learning with applications to various downstream tasks, the computational cost of training and inference remains a major challenge for applying these models to a wide spectrum of devices. Several isolated attempts have been made to compress Transformers, but the settings and metrics are different across studies. Trade-off at various compression rates are also largely missing in prior work, making it difficult to compare compression techniques. In this work, we aim to provide context for the isolated results, studying several commonly used compression techniques, including weight pruning, head pruning, low-rank approximation, and knowledge distillation. We report trade- off at various compression rate, including wall-clock time, the number of parameters, and the number of multiply-accumulate operations. Our results show that compared to recent approaches, basic compression techniques are strong baselines. We further present several
    

