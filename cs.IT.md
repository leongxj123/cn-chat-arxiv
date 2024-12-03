# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rotation Invariant Quantization for Model Compression.](http://arxiv.org/abs/2303.03106) | 本研究提出了一种旋转不变量量化（RIQ）技术，可以在不同层次上实现混合精度量化，用于后训练神经网络模型压缩，并证明了其在压缩方面的优势。在多种模型和任务上进行了严格评估，取得了令人满意的结果。 |

# 详细

[^1]: 旋转不变量量化用于模型压缩

    Rotation Invariant Quantization for Model Compression. (arXiv:2303.03106v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.03106](http://arxiv.org/abs/2303.03106)

    本研究提出了一种旋转不变量量化（RIQ）技术，可以在不同层次上实现混合精度量化，用于后训练神经网络模型压缩，并证明了其在压缩方面的优势。在多种模型和任务上进行了严格评估，取得了令人满意的结果。

    

    后训练神经网络（NN）模型压缩是一种将大型、消耗内存的模型部署到内存资源有限设备上的吸引人的方法。本研究探讨了NN模型压缩的速率-失真权衡。首先，我们提出了一种旋转不变量量化（RIQ）技术，它利用一个单一参数量化整个NN模型，在每个层次上得到不同的速率，即混合精度量化。然后，我们证明了我们的旋转不变量方法在压缩方面的优势。我们对RIQ进行了严格评估，并展示了它在各种模型和任务上的能力。例如，RIQ在预训练的VGG稠密和修剪模型上分别实现了19.4倍和52.9倍的压缩比，精度降低小于0.4%。代码可以在\url{https://github.com/ehaleva/RIQ}上找到。

    Post-training Neural Network (NN) model compression is an attractive approach for deploying large, memory-consuming models on devices with limited memory resources. In this study, we investigate the rate-distortion tradeoff for NN model compression. First, we suggest a Rotation-Invariant Quantization (RIQ) technique that utilizes a single parameter to quantize the entire NN model, yielding a different rate at each layer, i.e., mixed-precision quantization. Then, we prove that our rotation-invariant approach is optimal in terms of compression. We rigorously evaluate RIQ and demonstrate its capabilities on various models and tasks. For example, RIQ facilitates $\times 19.4$ and $\times 52.9$ compression ratios on pre-trained VGG dense and pruned models, respectively, with $<0.4\%$ accuracy degradation. Code is available in \url{https://github.com/ehaleva/RIQ}.
    

