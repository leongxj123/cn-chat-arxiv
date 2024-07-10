# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoChunk: Automated Activation Chunk for Memory-Efficient Long Sequence Inference.](http://arxiv.org/abs/2401.10652) | AutoChunk是一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。 |
| [^2] | [CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.](http://arxiv.org/abs/2308.15136) | CAGRA是一种面向GPU的高度并行图构建和近似最近邻搜索方法，在近似最近邻搜索领域取得了显著的效率提升。 |

# 详细

[^1]: AutoChunk: 自动激活块用于内存高效的长序列推断

    AutoChunk: Automated Activation Chunk for Memory-Efficient Long Sequence Inference. (arXiv:2401.10652v1 [cs.PF])

    [http://arxiv.org/abs/2401.10652](http://arxiv.org/abs/2401.10652)

    AutoChunk是一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。

    

    大型深度学习模型在各种应用中取得了令人瞩目的性能。然而，它们对内存的大量需求，包括参数内存和激活内存，已经成为实际应用中的重大挑战。现有方法主要处理参数内存，对激活内存的重要性却被忽视了。特别是对于长输入序列，随着序列长度的增加，激活内存预计会经历显著的指数增长。在这个方法中，我们提出了AutoChunk，一种自动和自适应的编译器系统，通过块策略有效地减少长序列推断的激活内存。所提出的系统通过多个阶段的优化生成块计划。在每个阶段，块搜索通过探索所有可能的块候选项，块选择通过识别最佳块进行。运行时，AutoChunk采用代码生成自动应用块策略。

    Large deep learning models have achieved impressive performance across a range of applications. However, their large memory requirements, including parameter memory and activation memory, have become a significant challenge for their practical serving. While existing methods mainly address parameter memory, the importance of activation memory has been overlooked. Especially for long input sequences, activation memory is expected to experience a significant exponential growth as the length of sequences increases. In this approach, we propose AutoChunk, an automatic and adaptive compiler system that efficiently reduces activation memory for long sequence inference by chunk strategies. The proposed system generates chunk plans by optimizing through multiple stages. In each stage, the chunk search pass explores all possible chunk candidates and the chunk selection pass identifies the optimal one. At runtime, AutoChunk employs code generation to automatically apply chunk strategies. The exp
    
[^2]: CAGRA：面向GPU的高度并行图构建和近似最近邻搜索

    CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs. (arXiv:2308.15136v1 [cs.DS])

    [http://arxiv.org/abs/2308.15136](http://arxiv.org/abs/2308.15136)

    CAGRA是一种面向GPU的高度并行图构建和近似最近邻搜索方法，在近似最近邻搜索领域取得了显著的效率提升。

    

    近似最近邻搜索（ANNS）在数据挖掘和人工智能领域中起着关键作用，涵盖了信息检索、计算机视觉、自然语言处理和推荐系统等各个学科。近年来，数据量急剧增加，穷举精确最近邻搜索的计算成本往往是禁止性的，必须采用近似技术。尽管图形化方法的平衡性能和召回率在ANNS算法中最近引起了广泛关注，但只有少数研究探索了利用GPU和多核处理器的强大计算能力，尽管广泛使用了大规模并行和通用计算能力。为了弥补这一差距，我们引入了一种基于并行计算硬件的新颖接近图和搜索算法。通过利用现代硬件的高性能能力，我们的方法实现了显著的效率提升。具体而言，我们的方法实现了高效的图构建和近似最近邻搜索。

    Approximate Nearest Neighbor Search (ANNS) plays a critical role in various disciplines spanning data mining and artificial intelligence, from information retrieval and computer vision to natural language processing and recommender systems. Data volumes have soared in recent years and the computational cost of an exhaustive exact nearest neighbor search is often prohibitive, necessitating the adoption of approximate techniques. The balanced performance and recall of graph-based approaches have more recently garnered significant attention in ANNS algorithms, however, only a few studies have explored harnessing the power of GPUs and multi-core processors despite the widespread use of massively parallel and general-purpose computing. To bridge this gap, we introduce a novel parallel computing hardware-based proximity graph and search algorithm. By leveraging the high-performance capabilities of modern hardware, our approach achieves remarkable efficiency gains. In particular, our method s
    

