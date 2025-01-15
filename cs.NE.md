# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Set-based Neural Network Encoding.](http://arxiv.org/abs/2305.16625) | 提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。 |

# 详细

[^1]: 集合化的神经网络编码

    Set-based Neural Network Encoding. (arXiv:2305.16625v1 [cs.LG])

    [http://arxiv.org/abs/2305.16625](http://arxiv.org/abs/2305.16625)

    提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。

    

    我们提出了一种利用集合到集合和集合到向量函数来有效编码神经网络参数，进行泛化性能预测的神经网络权重编码方法。与之前需要对不同架构编写自定义编码模型的方法不同，我们的方法能够对混合架构和不同参数大小的模型动态编码。此外，我们的 SNE（集合化神经网络编码器）通过使用一种逐层编码方案，考虑神经网络的分层计算结构。最终将所有层次编码合并到一起，以获取神经网络编码矢量。我们还引入了“pad-chunk-encode”流水线来有效地编码神经网络层，该流水线可根据计算和内存限制进行调整。我们还引入了两个用于神经网络泛化性能预测的新任务：跨数据集和架构适应性预测。

    We propose an approach to neural network weight encoding for generalization performance prediction that utilizes set-to-set and set-to-vector functions to efficiently encode neural network parameters. Our approach is capable of encoding neural networks in a modelzoo of mixed architecture and different parameter sizes as opposed to previous approaches that require custom encoding models for different architectures. Furthermore, our \textbf{S}et-based \textbf{N}eural network \textbf{E}ncoder (SNE) takes into consideration the hierarchical computational structure of neural networks by utilizing a layer-wise encoding scheme that culminates to encoding all layer-wise encodings to obtain the neural network encoding vector. Additionally, we introduce a \textit{pad-chunk-encode} pipeline to efficiently encode neural network layers that is adjustable to computational and memory constraints. We also introduce two new tasks for neural network generalization performance prediction: cross-dataset a
    

