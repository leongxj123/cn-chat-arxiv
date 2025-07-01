# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Spiking Neural Network Learning Methods with Varying Locality](https://arxiv.org/abs/2402.01782) | 本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。 |

# 详细

[^1]: 使用不同局部性对脉冲神经网络学习方法进行基准测试

    Benchmarking Spiking Neural Network Learning Methods with Varying Locality

    [https://arxiv.org/abs/2402.01782](https://arxiv.org/abs/2402.01782)

    本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。

    

    脉冲神经网络（SNN）提供更真实的神经动力学，在多个机器学习任务中已经显示出与人工神经网络（ANN）相当的性能。信息在SNN中以脉冲形式进行处理，采用事件驱动机制，显著降低了能源消耗。然而，由于脉冲机制的非可微性，训练SNN具有挑战性。传统方法如时间反向传播（BPTT）已经显示出一定的效果，但在计算和存储成本方面存在问题，并且在生物学上不可行。相反，最近的研究提出了具有不同局部性的替代学习方法，在分类任务中取得了成功。本文表明，这些方法在训练过程中有相似之处，同时在生物学合理性和性能之间存在权衡。此外，本研究还探讨了SNN的隐式循环特性，并进行了调查。

    Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but comes with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, this research examines the implicitly recurrent nature of SNNs and investigat
    

