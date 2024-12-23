# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Residual Multi-Fidelity Neural Network Computing.](http://arxiv.org/abs/2310.03572) | 本研究提出了一种残差多保真计算框架，通过使用多保真信息构建神经网络代理模型，解决了低保真和高保真计算模型之间的相关性建模问题。这种方法训练了两个神经网络，利用残差函数进行模型训练，最终得到了高保真替代模型。 |

# 详细

[^1]: 多保真神经网络计算的残差方法

    Residual Multi-Fidelity Neural Network Computing. (arXiv:2310.03572v1 [cs.LG])

    [http://arxiv.org/abs/2310.03572](http://arxiv.org/abs/2310.03572)

    本研究提出了一种残差多保真计算框架，通过使用多保真信息构建神经网络代理模型，解决了低保真和高保真计算模型之间的相关性建模问题。这种方法训练了两个神经网络，利用残差函数进行模型训练，最终得到了高保真替代模型。

    

    在本研究中，我们考虑使用多保真信息构建神经网络代理模型的一般问题。给定一个廉价的低保真和一个昂贵的高保真计算模型，我们提出了一个残差多保真计算框架，将模型之间的相关性建模为一个残差函数，这是一个可能非线性的1）模型共享的输入空间和低保真模型输出之间的映射，以及2）两个模型输出之间的差异。为了实现这一点，我们训练了两个神经网络来协同工作。第一个网络在少量的高保真和低保真数据上学习残差函数。一旦训练完成，这个网络被用来生成额外的合成高保真数据，用于训练第二个网络。一旦训练完成，第二个网络作为我们对高保真感兴趣的量的替代模型。我们提供了三个数值例子来证明这种方法的能力。

    In this work, we consider the general problem of constructing a neural network surrogate model using multi-fidelity information. Given an inexpensive low-fidelity and an expensive high-fidelity computational model, we present a residual multi-fidelity computational framework that formulates the correlation between models as a residual function, a possibly non-linear mapping between 1) the shared input space of the models together with the low-fidelity model output and 2) the discrepancy between the two model outputs. To accomplish this, we train two neural networks to work in concert. The first network learns the residual function on a small set of high-fidelity and low-fidelity data. Once trained, this network is used to generate additional synthetic high-fidelity data, which is used in the training of a second network. This second network, once trained, acts as our surrogate for the high-fidelity quantity of interest. We present three numerical examples to demonstrate the power of th
    

