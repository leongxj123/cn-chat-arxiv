# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks.](http://arxiv.org/abs/2209.11740) | 本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。 |
| [^2] | [Energy-Latency Attacks via Sponge Poisoning.](http://arxiv.org/abs/2203.08147) | 本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。 |

# 详细

[^1]: 关于卷积神经网络中最大池化特征图的位移不变性

    On the Shift Invariance of Max Pooling Feature Maps in Convolutional Neural Networks. (arXiv:2209.11740v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.11740](http://arxiv.org/abs/2209.11740)

    本文研究了卷积神经网络中最大池化特征图的位移不变性问题，并提出了一种近似复数模的条件，实现了位移稳定性。实验证实了理论的有效性。

    

    本文致力于改善卷积神经网络（CNN）在图像分类领域中的数学可解释性。具体而言，我们解决了在其第一层中出现的不稳定性问题。当在像ImageNet这样的数据集上进行训练时，其第一层往往学习到与方向边通滤波器非常相似的参数。使用这样的Gabor滤波器进行子采样卷积容易出现混叠问题，导致对输入的小偏移敏感。在这个背景下，我们建立了最大池化算子近似复数模的条件，使其几乎具有位移不变性。然后，我们推导了子采样卷积后最大池化的位移稳定性度量。特别地，我们强调了滤波器的频率和方向在实现稳定性方面的关键作用。通过考虑基于双树复小波包变换的确定性特征提取器，即离散Gabor的一种特殊情况，我们通过实验证实了我们的理论。

    This paper focuses on improving the mathematical interpretability of convolutional neural networks (CNNs) in the context of image classification. Specifically, we tackle the instability issue arising in their first layer, which tends to learn parameters that closely resemble oriented band-pass filters when trained on datasets like ImageNet. Subsampled convolutions with such Gabor-like filters are prone to aliasing, causing sensitivity to small input shifts. In this context, we establish conditions under which the max pooling operator approximates a complex modulus, which is nearly shift invariant. We then derive a measure of shift invariance for subsampled convolutions followed by max pooling. In particular, we highlight the crucial role played by the filter's frequency and orientation in achieving stability. We experimentally validate our theory by considering a deterministic feature extractor based on the dual-tree complex wavelet packet transform, a particular case of discrete Gabor
    
[^2]: 基于海绵毒化的能耗延迟攻击。

    Energy-Latency Attacks via Sponge Poisoning. (arXiv:2203.08147v4 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2203.08147](http://arxiv.org/abs/2203.08147)

    本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。

    

    海绵样本是在测试时精心优化的输入，可在硬件加速器上部署时增加神经网络的能量消耗和延迟。本文首次证明了海绵样本也可通过一种名为海绵毒化的攻击注入到训练中。该攻击允许在每个测试时输入中不加区分地提高机器学习模型的能量消耗和延迟。我们提出了一种新的海绵毒化形式化方法，克服了与优化测试时海绵样本相关的限制，并表明即使攻击者仅控制几个模型更新，例如模型训练被外包给不受信任的第三方或通过联邦学习分布式进行，也可以进行这种攻击。我们进行了广泛的实验分析，表明海绵毒化几乎完全消除了硬件加速器的效果。同时，我们还分析了毒化模型的激活，确定了哪些计算对导致能量消耗和延迟增加起重要作用。

    Sponge examples are test-time inputs carefully optimized to increase energy consumption and latency of neural networks when deployed on hardware accelerators. In this work, we are the first to demonstrate that sponge examples can also be injected at training time, via an attack that we call sponge poisoning. This attack allows one to increase the energy consumption and latency of machine-learning models indiscriminately on each test-time input. We present a novel formalization for sponge poisoning, overcoming the limitations related to the optimization of test-time sponge examples, and show that this attack is possible even if the attacker only controls a few model updates; for instance, if model training is outsourced to an untrusted third-party or distributed via federated learning. Our extensive experimental analysis shows that sponge poisoning can almost completely vanish the effect of hardware accelerators. We also analyze the activations of poisoned models, identifying which comp
    

