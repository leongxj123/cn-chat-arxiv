# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GCondNet: A Novel Method for Improving Neural Networks on Small High-Dimensional Tabular Data.](http://arxiv.org/abs/2211.06302) | GCondNet利用高维表格数据的隐含结构，通过创建图形并利用图神经网络以及条件训练，提高了潜在预测网络的性能。 |

# 详细

[^1]: GCondNet: 一种改进小型高维表格数据神经网络的新方法

    GCondNet: A Novel Method for Improving Neural Networks on Small High-Dimensional Tabular Data. (arXiv:2211.06302v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.06302](http://arxiv.org/abs/2211.06302)

    GCondNet利用高维表格数据的隐含结构，通过创建图形并利用图神经网络以及条件训练，提高了潜在预测网络的性能。

    

    神经网络模型在处理高维但样本数量较小的表格数据集时经常遇到困难。其中一个原因是当前的权重初始化方法假定权重之间相互独立，当样本不足以准确估计模型参数时，这可能会产生问题。在这种小数据场景下，利用其他结构可以提高模型的训练稳定性和性能。为解决这个问题，我们提出了GCondNet，一种通过利用表格数据中的隐含结构来增强神经网络的通用方法。我们针对每个数据维度在样本之间创建一个图形，并利用图神经网络 (GNN) 提取这种隐含结构，以及调整潜在预测 MLP 网络的第一层参数进行条件训练。通过创建许多小图，GCondNet 利用了数据的高维特性，从而提高了潜在预测网络的性能。我们通过实验证明了我们的方法的有效性。

    Neural network models often struggle with high-dimensional but small sample-size tabular datasets. One reason is that current weight initialisation methods assume independence between weights, which can be problematic when there are insufficient samples to estimate the model's parameters accurately. In such small data scenarios, leveraging additional structures can improve the model's training stability and performance. To address this, we propose GCondNet, a general approach to enhance neural networks by leveraging implicit structures present in tabular data. We create a graph between samples for each data dimension, and utilise Graph Neural Networks (GNNs) for extracting this implicit structure, and for conditioning the parameters of the first layer of an underlying predictor MLP network. By creating many small graphs, GCondNet exploits the data's high-dimensionality, and thus improves the performance of an underlying predictor network. We demonstrate the effectiveness of our method 
    

