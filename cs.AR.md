# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerating Graph Neural Networks on Real Processing-In-Memory Systems](https://arxiv.org/abs/2402.16731) | 在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。 |
| [^2] | [A Cost-Efficient FPGA Implementation of Tiny Transformer Model using Neural ODE.](http://arxiv.org/abs/2401.02721) | 本文提出了一种利用神经ODE作为骨干架构的高性价比FPGA实现微型Transformer模型。该模型相比于基于CNN的模型将参数大小减少了94.6%且保持准确性，适用于边缘计算。 |

# 详细

[^1]: 在实际处理内存系统上加速图神经网络

    Accelerating Graph Neural Networks on Real Processing-In-Memory Systems

    [https://arxiv.org/abs/2402.16731](https://arxiv.org/abs/2402.16731)

    在实际处理内存系统上加速图神经网络，并提出了针对实际PIM系统的智能并行化技术和混合式执行方法。

    

    图神经网络（GNNs）是新兴的机器学习模型，用于分析图结构数据。图神经网络（GNN）的执行涉及计算密集型和内存密集型核心，后者在总时间中占主导地位，受数据在内存和处理器之间移动的严重瓶颈所限制。处理内存（PIM）系统可以通过在内存阵列附近或内部放置简单处理器来缓解这种数据移动瓶颈。在这项工作中，我们介绍了PyGim，一个有效的机器学习框架，可以在实际PIM系统上加速GNNs。我们为针对实际PIM系统定制的GNN内存密集型核心提出智能并行化技术，并为它们开发了方便的Python API。我们提供混合式GNN执行，其中计算密集型和内存密集型核心分别在以处理器为中心和以内存为中心的计算系统中执行，以匹配它们的算法特性。我们进行了大量评估。

    arXiv:2402.16731v2 Announce Type: replace-cross  Abstract: Graph Neural Networks (GNNs) are emerging ML models to analyze graph-structure data. Graph Neural Network (GNN) execution involves both compute-intensive and memory-intensive kernels, the latter dominates the total time, being significantly bottlenecked by data movement between memory and processors. Processing-In-Memory (PIM) systems can alleviate this data movement bottleneck by placing simple processors near or inside to memory arrays. In this work, we introduce PyGim, an efficient ML framework that accelerates GNNs on real PIM systems. We propose intelligent parallelization techniques for memory-intensive kernels of GNNs tailored for real PIM systems, and develop handy Python API for them. We provide hybrid GNN execution, in which the compute-intensive and memory-intensive kernels are executed in processor-centric and memory-centric computing systems, respectively, to match their algorithmic nature. We extensively evaluate 
    
[^2]: 利用神经ODE的高性价比FPGA实现微型Transformer模型

    A Cost-Efficient FPGA Implementation of Tiny Transformer Model using Neural ODE. (arXiv:2401.02721v1 [cs.LG])

    [http://arxiv.org/abs/2401.02721](http://arxiv.org/abs/2401.02721)

    本文提出了一种利用神经ODE作为骨干架构的高性价比FPGA实现微型Transformer模型。该模型相比于基于CNN的模型将参数大小减少了94.6%且保持准确性，适用于边缘计算。

    

    Transformer是一种具有注意机制的新兴神经网络模型。它已经被用于各种任务，并且相比于CNN和RNN取得了良好的准确性。虽然注意机制被认为是一种通用的组件，但是许多Transformer模型与基于CNN的模型相比需要大量的参数。为了减少计算复杂性，最近提出了一种混合方法，它使用ResNet作为骨干架构，并将部分卷积层替换为MHSA（多头自注意）机制。在本文中，我们通过使用神经ODE（常微分方程）而不是ResNet作为骨干架构，显著减少了这种模型的参数大小。所提出的混合模型相比于基于CNN的模型将参数大小减少了94.6%，而且没有降低准确性。接着，我们将所提出的模型部署在一台适度规模的FPGA设备上进行边缘计算。

    Transformer is an emerging neural network model with attention mechanism. It has been adopted to various tasks and achieved a favorable accuracy compared to CNNs and RNNs. While the attention mechanism is recognized as a general-purpose component, many of the Transformer models require a significant number of parameters compared to the CNN-based ones. To mitigate the computational complexity, recently, a hybrid approach has been proposed, which uses ResNet as a backbone architecture and replaces a part of its convolution layers with an MHSA (Multi-Head Self-Attention) mechanism. In this paper, we significantly reduce the parameter size of such models by using Neural ODE (Ordinary Differential Equation) as a backbone architecture instead of ResNet. The proposed hybrid model reduces the parameter size by 94.6% compared to the CNN-based ones without degrading the accuracy. We then deploy the proposed model on a modest-sized FPGA device for edge computing. To further reduce FPGA resource u
    

