# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuralFuse: Learning to Improve the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes.](http://arxiv.org/abs/2306.16869) | NeuralFuse是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，解决了低电压环境下有限访问神经网络推断的准确性与能量之间的权衡问题。 |

# 详细

[^1]: NeuralFuse: 学习改善低电压环境下有限访问神经网络推断的准确性

    NeuralFuse: Learning to Improve the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes. (arXiv:2306.16869v1 [cs.LG])

    [http://arxiv.org/abs/2306.16869](http://arxiv.org/abs/2306.16869)

    NeuralFuse是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，解决了低电压环境下有限访问神经网络推断的准确性与能量之间的权衡问题。

    

    深度神经网络在机器学习中已经无处不在，但其能量消耗仍然是一个值得关注的问题。降低供电电压是降低能量消耗的有效策略。然而，过度降低供电电压可能会导致准确性降低，因为模型参数存储在静态随机存储器(SRAM)中，而SRAM中会发生随机位翻转。为了解决这个挑战，我们引入了NeuralFuse，这是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，以在低电压环境中解决准确性与能量之间的权衡。NeuralFuse在标称电压和低电压情况下都能保护DNN的准确性。此外，NeuralFuse易于实现，并可以轻松应用于有限访问的DNN，例如不可配置的硬件或云端API的远程访问。实验结果表明，在1%的位错误率下，NeuralFuse可以将SRAM内存访问能量降低高达24%，同时保持准确性。

    Deep neural networks (DNNs) have become ubiquitous in machine learning, but their energy consumption remains a notable issue. Lowering the supply voltage is an effective strategy for reducing energy consumption. However, aggressively scaling down the supply voltage can lead to accuracy degradation due to random bit flips in static random access memory (SRAM) where model parameters are stored. To address this challenge, we introduce NeuralFuse, a novel add-on module that addresses the accuracy-energy tradeoff in low-voltage regimes by learning input transformations to generate error-resistant data representations. NeuralFuse protects DNN accuracy in both nominal and low-voltage scenarios. Moreover, NeuralFuse is easy to implement and can be readily applied to DNNs with limited access, such as non-configurable hardware or remote access to cloud-based APIs. Experimental results demonstrate that, at a 1% bit error rate, NeuralFuse can reduce SRAM memory access energy by up to 24% while imp
    

