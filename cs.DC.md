# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Graph Neural Networks on Real Processing-In-Memory Systems](https://arxiv.org/abs/2402.16731) | 在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。 |

# 详细

[^1]: 在实际处理内存系统上加速图神经网络

    Accelerating Graph Neural Networks on Real Processing-In-Memory Systems

    [https://arxiv.org/abs/2402.16731](https://arxiv.org/abs/2402.16731)

    在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。

    

    图神经网络（GNNs）是新兴的机器学习模型，用于分析图结构数据。图神经网络（GNN）的执行涉及计算密集型和内存密集型核心，后者在总时间中占主导地位，受数据在内存和处理器之间移动的严重瓶颈所限制。处理内存（PIM）系统可以通过在内存阵列附近或内部放置简单处理器来缓解这种数据移动瓶颈。在这项工作中，我们介绍了PyGim，一个有效的机器学习框架，可以在实际PIM系统上加速GNNs。我们为针对实际PIM系统定制的GNN内存密集型核心提出智能并行化技术，并为它们开发了方便的Python API。我们提供混合式GNN执行，其中计算密集型和内存密集型核心分别在以处理器为中心和以内存为中心的计算系统中执行，以匹配它们的算法特性。我们进行了大量评估。

    arXiv:2402.16731v2 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNNs) are emerging ML models to analyze graph-structure data. Graph Neural Network (GNN) execution involves both compute-intensive and memory-intensive kernels, the latter dominates the total time, being significantly bottlenecked by data movement between memory and processors. Processing-In-Memory (PIM) systems can alleviate this data movement bottleneck by placing simple processors near or inside to memory arrays. In this work, we introduce PyGim, an efficient ML framework that accelerates GNNs on real PIM systems. We propose intelligent parallelization techniques for memory-intensive kernels of GNNs tailored for real PIM systems, and develop handy Python API for them. We provide hybrid GNN execution, in which the compute-intensive and memory-intensive kernels are executed in processor-centric and memory-centric computing systems, respectively, to match their algorithmic nature. We extensively evaluate 
    

