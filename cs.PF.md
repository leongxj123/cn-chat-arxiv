# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Graph Neural Networks on Real Processing-In-Memory Systems](https://arxiv.org/abs/2402.16731) | 在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。 |
| [^2] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |

# 详细

[^1]: 在实际处理内存系统上加速图神经网络

    Accelerating Graph Neural Networks on Real Processing-In-Memory Systems

    [https://arxiv.org/abs/2402.16731](https://arxiv.org/abs/2402.16731)

    在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。

    

    图神经网络（GNNs）是新兴的机器学习模型，用于分析图结构数据。图神经网络（GNN）的执行涉及计算密集型和内存密集型核心，后者在总时间中占主导地位，受数据在内存和处理器之间移动的严重瓶颈所限制。处理内存（PIM）系统可以通过在内存阵列附近或内部放置简单处理器来缓解这种数据移动瓶颈。在这项工作中，我们介绍了PyGim，一个有效的机器学习框架，可以在实际PIM系统上加速GNNs。我们为针对实际PIM系统定制的GNN内存密集型核心提出智能并行化技术，并为它们开发了方便的Python API。我们提供混合式GNN执行，其中计算密集型和内存密集型核心分别在以处理器为中心和以内存为中心的计算系统中执行，以匹配它们的算法特性。我们进行了大量评估。

    arXiv:2402.16731v2 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNNs) are emerging ML models to analyze graph-structure data. Graph Neural Network (GNN) execution involves both compute-intensive and memory-intensive kernels, the latter dominates the total time, being significantly bottlenecked by data movement between memory and processors. Processing-In-Memory (PIM) systems can alleviate this data movement bottleneck by placing simple processors near or inside to memory arrays. In this work, we introduce PyGim, an efficient ML framework that accelerates GNNs on real PIM systems. We propose intelligent parallelization techniques for memory-intensive kernels of GNNs tailored for real PIM systems, and develop handy Python API for them. We provide hybrid GNN execution, in which the compute-intensive and memory-intensive kernels are executed in processor-centric and memory-centric computing systems, respectively, to match their algorithmic nature. We extensively evaluate 
    
[^2]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    

