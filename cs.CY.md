# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Censoring chemical data to mitigate dual use risk.](http://arxiv.org/abs/2304.10510) | 为了缓解化学数据被用于开发识别新型毒素或化学战剂的预测模型的双重用途风险，提出了一种基于模型不可知的方法，可在保留有利于训练深度神经网络的数据的效用区域的同时，对敏感信息的数据集进行有选择性的添加噪音，该方法被证明是有效的，并且忽略敏感数据会增加模型的方差。 |

# 详细

[^1]: 缓解双重用途风险的化学数据审查方法

    Censoring chemical data to mitigate dual use risk. (arXiv:2304.10510v1 [cs.LG])

    [http://arxiv.org/abs/2304.10510](http://arxiv.org/abs/2304.10510)

    为了缓解化学数据被用于开发识别新型毒素或化学战剂的预测模型的双重用途风险，提出了一种基于模型不可知的方法，可在保留有利于训练深度神经网络的数据的效用区域的同时，对敏感信息的数据集进行有选择性的添加噪音，该方法被证明是有效的，并且忽略敏感数据会增加模型的方差。

    

    机器学习应用的双重用途（模型既可以用于有益用途又可以用于恶意目的）提出了重大挑战。最近，在化学领域，由于含有敏感标签（如毒理学信息）的化学数据集可能被用于开发识别新型毒素或化学战剂的预测模型，这已成为一个特别的担忧。为了缓解双重用途风险，我们提出了一种基于模型不可知的方法，可以有选择性地添加噪音到数据集中，同时保留有利于训练深度神经网络的数据的效用区域。我们评估了所提出方法在最小二乘法、多层感知器和图神经网络中的有效性, 发现有选择性添加噪声数据可以引入模型的方差和预测敏感标签的偏差，这表明可以实现包含敏感信息的数据集的安全共享。我们还发现，忽略敏感数据通常会增加模型的方差。

    The dual use of machine learning applications, where models can be used for both beneficial and malicious purposes, presents a significant challenge. This has recently become a particular concern in chemistry, where chemical datasets containing sensitive labels (e.g. toxicological information) could be used to develop predictive models that identify novel toxins or chemical warfare agents. To mitigate dual use risks, we propose a model-agnostic method of selectively noising datasets while preserving the utility of the data for training deep neural networks in a beneficial region. We evaluate the effectiveness of the proposed method across least squares, a multilayer perceptron, and a graph neural network. Our findings show selectively noised datasets can induce model variance and bias in predictions for sensitive labels with control, suggesting the safe sharing of datasets containing sensitive information is feasible. We also find omitting sensitive data often increases model variance 
    

