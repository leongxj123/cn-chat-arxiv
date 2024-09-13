# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems](https://arxiv.org/abs/2403.00381) | 提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。 |
| [^2] | [Short-term power load forecasting method based on CNN-SAEDN-Res.](http://arxiv.org/abs/2309.07140) | 提出了一种基于CNN-SAEDN-Res的短期功率负荷预测方法，通过结合卷积神经网络、自注意力编码器-解码器网络和残差修正技术，能够有效处理带有非时序因素的负荷数据并提高预测精度。 |

# 详细

[^1]: 基于结构化深度神经网络的拉格朗日系统反步轨迹跟踪控制

    Structured Deep Neural Networks-Based Backstepping Trajectory Tracking Control for Lagrangian Systems

    [https://arxiv.org/abs/2403.00381](https://arxiv.org/abs/2403.00381)

    提出了一种基于结构化DNN的控制器，通过设计神经网络结构确保闭环稳定性，并进一步优化参数以实现改进的控制性能，同时提供了关于跟踪误差的明确上限。

    

    深度神经网络（DNN）越来越多地被用于学习控制器，因为其出色的逼近能力。然而，它们的黑盒特性对闭环稳定性保证和性能分析构成了重要挑战。在本文中，我们引入了一种基于结构化DNN的控制器，用于采用反推技术实现拉格朗日系统的轨迹跟踪控制。通过适当设计神经网络结构，所提出的控制器可以确保任何兼容的神经网络参数实现闭环稳定性。此外，通过进一步优化神经网络参数，可以实现更好的控制性能。此外，我们提供了关于跟踪误差的明确上限，这允许我们通过适当选择控制参数来实现所需的跟踪性能。此外，当系统模型未知时，我们提出了一种改进的拉格朗日神经网络。

    arXiv:2403.00381v1 Announce Type: cross  Abstract: Deep neural networks (DNN) are increasingly being used to learn controllers due to their excellent approximation capabilities. However, their black-box nature poses significant challenges to closed-loop stability guarantees and performance analysis. In this paper, we introduce a structured DNN-based controller for the trajectory tracking control of Lagrangian systems using backing techniques. By properly designing neural network structures, the proposed controller can ensure closed-loop stability for any compatible neural network parameters. In addition, improved control performance can be achieved by further optimizing neural network parameters. Besides, we provide explicit upper bounds on tracking errors in terms of controller parameters, which allows us to achieve the desired tracking performance by properly selecting the controller parameters. Furthermore, when system models are unknown, we propose an improved Lagrangian neural net
    
[^2]: 基于CNN-SAEDN-Res的短期功率负荷预测方法

    Short-term power load forecasting method based on CNN-SAEDN-Res. (arXiv:2309.07140v1 [eess.SP])

    [http://arxiv.org/abs/2309.07140](http://arxiv.org/abs/2309.07140)

    提出了一种基于CNN-SAEDN-Res的短期功率负荷预测方法，通过结合卷积神经网络、自注意力编码器-解码器网络和残差修正技术，能够有效处理带有非时序因素的负荷数据并提高预测精度。

    

    在深度学习中，带有非时序因素的负荷数据难以通过序列模型进行处理。这个问题导致了预测的精度不足。因此，提出了一种基于卷积神经网络（CNN）、自注意力编码器-解码器网络（SAEDN）和残差修正（Res）的短期负荷预测方法。在该方法中，特征提取模块由一个二维卷积神经网络组成，用于挖掘数据之间的局部相关性并获取高维数据特征。初始负荷预测模块由一个自注意力编码器-解码器网络和一个前馈神经网络（FFN）组成。该模块利用自注意机制对高维特征进行编码。这个操作可以获取数据之间的全局相关性。因此，该模型能够基于混合了非时序因素的数据中的耦合关系保留重要的信息。

    In deep learning, the load data with non-temporal factors are difficult to process by sequence models. This problem results in insufficient precision of the prediction. Therefore, a short-term load forecasting method based on convolutional neural network (CNN), self-attention encoder-decoder network (SAEDN) and residual-refinement (Res) is proposed. In this method, feature extraction module is composed of a two-dimensional convolutional neural network, which is used to mine the local correlation between data and obtain high-dimensional data features. The initial load fore-casting module consists of a self-attention encoder-decoder network and a feedforward neural network (FFN). The module utilizes self-attention mechanisms to encode high-dimensional features. This operation can obtain the global correlation between data. Therefore, the model is able to retain important information based on the coupling relationship between the data in data mixed with non-time series factors. Then, self
    

