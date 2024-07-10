# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoChunk: Automated Activation Chunk for Memory-Efficient Long Sequence Inference.](http://arxiv.org/abs/2401.10652) | AutoChunk是一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。 |

# 详细

[^1]: AutoChunk: 自动激活块用于内存高效的长序列推断

    AutoChunk: Automated Activation Chunk for Memory-Efficient Long Sequence Inference. (arXiv:2401.10652v1 [cs.PF])

    [http://arxiv.org/abs/2401.10652](http://arxiv.org/abs/2401.10652)

    AutoChunk是一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。

    

    大型深度学习模型在各种应用中取得了令人瞩目的性能。然而，它们对内存的大量需求，包括参数内存和激活内存，已经成为实际应用中的重大挑战。现有方法主要处理参数内存，对激活内存的重要性却被忽视了。特别是对于长输入序列，随着序列长度的增加，激活内存预计会经历显著的指数增长。在这个方法中，我们提出了AutoChunk，一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。所提出的系统通过多个阶段的优化生成块计划。在每个阶段，块搜索通过探索所有可能的块候选项，块选择通过识别最佳块进行。运行时，AutoChunk采用代码生成自动应用块策略。

    Large deep learning models have achieved impressive performance across a range of applications. However, their large memory requirements, including parameter memory and activation memory, have become a significant challenge for their practical serving. While existing methods mainly address parameter memory, the importance of activation memory has been overlooked. Especially for long input sequences, activation memory is expected to experience a significant exponential growth as the length of sequences increases. In this approach, we propose AutoChunk, an automatic and adaptive compiler system that efficiently reduces activation memory for long sequence inference by chunk strategies. The proposed system generates chunk plans by optimizing through multiple stages. In each stage, the chunk search pass explores all possible chunk candidates and the chunk selection pass identifies the optimal one. At runtime, AutoChunk employs code generation to automatically apply chunk strategies. The exp
    

