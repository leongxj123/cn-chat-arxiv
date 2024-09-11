# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [INFLECT-DGNN: Influencer Prediction with Dynamic Graph Neural Networks.](http://arxiv.org/abs/2307.08131) | INFLECT-DGNN是一个结合了图神经网络和递归神经网络的框架，使用加权损失函数、针对图数据适应的合成少数过采样技术和滚动窗口策略，用于影响者预测。实验结果显示，使用RNN来编码时间属性和GNN显著提高了预测性能。 |

# 详细

[^1]: INFLECT-DGNN: 动态图神经网络进行影响者预测

    INFLECT-DGNN: Influencer Prediction with Dynamic Graph Neural Networks. (arXiv:2307.08131v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2307.08131](http://arxiv.org/abs/2307.08131)

    INFLECT-DGNN是一个结合了图神经网络和递归神经网络的框架，使用加权损失函数、针对图数据适应的合成少数过采样技术和滚动窗口策略，用于影响者预测。实验结果显示，使用RNN来编码时间属性和GNN显著提高了预测性能。

    

    在许多领域中，利用网络信息进行预测建模已经变得非常普遍。在推荐和定向营销领域中，影响者检测是一个可以通过动态网络表示大大受益的领域，原因是不断发展的客户-品牌关系。为了阐述这一思想，我们引入了INFLECT-DGNN，这是一个使用加权损失函数的图神经网络（GNN）和递归神经网络（RNN），针对图数据适应的合成少数过采样技术（SMOTE）以及精心设计的滚动窗口策略的新框架。为了评估预测性能，我们利用一个包含三个城市网络的独特企业数据集，并制定了一个以利润为驱动的影响者预测评估方法。我们的结果表明，使用RNN来编码时间属性以及GNN显著改善了预测性能。

    Leveraging network information for predictive modeling has become widespread in many domains. Within the realm of referral and targeted marketing, influencer detection stands out as an area that could greatly benefit from the incorporation of dynamic network representation due to the ongoing development of customer-brand relationships. To elaborate this idea, we introduce INFLECT-DGNN, a new framework for INFLuencer prEdiCTion with Dynamic Graph Neural Networks that combines Graph Neural Networks (GNN) and Recurrent Neural Networks (RNN) with weighted loss functions, the Synthetic Minority Oversampling TEchnique (SMOTE) adapted for graph data, and a carefully crafted rolling-window strategy. To evaluate predictive performance, we utilize a unique corporate data set with networks of three cities and derive a profit-driven evaluation methodology for influencer prediction. Our results show how using RNN to encode temporal attributes alongside GNNs significantly improves predictive perform
    

