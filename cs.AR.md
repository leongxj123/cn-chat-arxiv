# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Abstracting Sparse DNN Acceleration via Structured Sparse Tensor Decomposition](https://arxiv.org/abs/2403.07953) | 本文提出了通过结构化分解张量进一步抽象稀疏DNN加速的方法，实现了将稀疏张量转换成一系列结构化稀疏张量，从而弥合了稀疏DNN模型和硬件之间的差距。 |

# 详细

[^1]: 通过结构化稀疏张量分解对稀疏DNN加速进行抽象化

    Abstracting Sparse DNN Acceleration via Structured Sparse Tensor Decomposition

    [https://arxiv.org/abs/2403.07953](https://arxiv.org/abs/2403.07953)

    本文提出了通过结构化分解张量进一步抽象稀疏DNN加速的方法，实现了将稀疏张量转换成一系列结构化稀疏张量，从而弥合了稀疏DNN模型和硬件之间的差距。

    

    在深度神经网络（DNNs）中利用稀疏性已成为满足现代DNN日益增长的计算需求的一种具有前景的领域。然而，在实践中，稀疏DNN加速仍然面临一个关键挑战。为了最小化稀疏加速的开销，硬件设计师最近提出了结构化稀疏硬件支持，这提供了有限的灵活性并需要额外的模型微调。此外，为某些结构化稀疏硬件微调的任何稀疏模型无法被其他结构化硬件加速。为了弥合稀疏DNN模型和硬件之间的差距，本文提出了通过结构分解的张量近似（TASD），利用了线性代数中的分配性质将任何稀疏张量转化为一系列结构化稀疏张量。接下来，我们开发了一个软件框架TASDER，通过搜索逐层高质量的结构化分解来加速DNNs的权重和...

    arXiv:2403.07953v1 Announce Type: cross  Abstract: Exploiting sparsity in deep neural networks (DNNs) has been a promising area to meet the growing computation need of modern DNNs. However, in practice, sparse DNN acceleration still faces a key challenge. To minimize the overhead of sparse acceleration, hardware designers have proposed structured sparse hardware support recently, which provides limited flexibility and requires extra model fine-tuning. Moreover, any sparse model fine-tuned for certain structured sparse hardware cannot be accelerated by other structured hardware. To bridge the gap between sparse DNN models and hardware, this paper proposes tensor approximation via structured decomposition (TASD), which leverages the distributive property in linear algebra to turn any sparse tensor into a series of structured sparse tensors. Next, we develop a software framework, TASDER, to accelerate DNNs by searching layer-wise, high-quality structured decomposition for both weight and 
    

