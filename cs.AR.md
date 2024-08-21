# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuralMatrix: Moving Entire Neural Networks to General Matrix Multiplication for Efficient Inference.](http://arxiv.org/abs/2305.14405) | NeuralMatrix是一种框架，能够在单个通用矩阵乘法加速器上计算深度神经网络(DNNs)，并可在保持推理准确度的情况下实现高达113倍至19.44倍的性能提升。 |

# 详细

[^1]: NeuralMatrix: 将整个神经网络移动到通用矩阵乘法以实现高效推理

    NeuralMatrix: Moving Entire Neural Networks to General Matrix Multiplication for Efficient Inference. (arXiv:2305.14405v1 [cs.LG])

    [http://arxiv.org/abs/2305.14405](http://arxiv.org/abs/2305.14405)

    NeuralMatrix是一种框架，能够在单个通用矩阵乘法加速器上计算深度神经网络(DNNs)，并可在保持推理准确度的情况下实现高达113倍至19.44倍的性能提升。

    

    本研究介绍了一种名为NeuralMatrix的新型框架，它使得可以在单个通用矩阵乘法（GEMM）加速器上计算多功能的深度神经网络（DNNs）。该方法克服了基于ASIC的加速器的专用性限制，同时实现了与CPU和GPU等通用处理器相比的应用特定加速水平。我们解决了将DNN计算中的线性和非线性运算映射到通用矩阵乘法以及使用GEMM加速器对DNN推理准确性的影响的挑战。我们在来自三种流行类别的各种DNN模型上进行了大量实验（即CNN，Transformers和GNN）作为示例的支撑模型。我们的结果表明，将DNN转换为通用矩阵乘法后仅会出现高达2.02％的准确度损失，同时将吞吐量与功率的比值与CPU和GPU相比提高了113倍到19.44倍。

    In this study, we introduce NeuralMatrix, a novel framework that enables the computation of versatile deep neural networks (DNNs) on a single general matrix multiplication (GEMM) accelerator. The proposed approach overcomes the specificity limitations of ASIC-based accelerators while achieving application-specific acceleration levels compared to general-purpose processors such as CPUs and GPUs. We address the challenges of mapping both linear and nonlinear operations in DNN computation to general matrix multiplications and the impact of using a GEMM accelerator on DNN inference accuracy. Extensive experiments are conducted on various DNN models from three popular categories (i.e., CNN, Transformers, and GNN) as illustrative backbone models. Our results demonstrate that DNNs suffer only up to a 2.02% accuracy loss after being converted to general matrix multiplication, while achieving 113x to 19.44x improvements in throughput per power compared to CPUs and GPUs.
    

