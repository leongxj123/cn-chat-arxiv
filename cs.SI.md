# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A performance characteristic curve for model evaluation: the application in information diffusion prediction.](http://arxiv.org/abs/2309.09537) | 本研究提出了一种模型的性能特征曲线，用于评估其在不同复杂度任务中的表现。通过使用基于信息熵的度量方法，我们确定了随机性与模型预测准确性之间的关系，并发现不同条件下的数据点都可以合并成一条曲线，捕捉了模型在面对不确定性时的正确预测能力。 |
| [^2] | [Clarify Confused Nodes Through Separated Learning.](http://arxiv.org/abs/2306.02285) | 本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。 |

# 详细

[^1]: 一种模型评估的性能特征曲线：在信息扩散预测中的应用

    A performance characteristic curve for model evaluation: the application in information diffusion prediction. (arXiv:2309.09537v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2309.09537](http://arxiv.org/abs/2309.09537)

    本研究提出了一种模型的性能特征曲线，用于评估其在不同复杂度任务中的表现。通过使用基于信息熵的度量方法，我们确定了随机性与模型预测准确性之间的关系，并发现不同条件下的数据点都可以合并成一条曲线，捕捉了模型在面对不确定性时的正确预测能力。

    

    社交网络上的信息扩散预测旨在预测未来消息的接收者，在市场营销和社交媒体等实际应用中具有实用价值。尽管不同的预测模型都声称表现良好，但性能评估的通用框架仍然有限。本文旨在识别模型的性能特征曲线，该曲线捕获了模型在不同复杂度任务上的表现。我们提出了一种基于信息熵的度量方法来量化扩散数据中的随机性，然后确定了随机性与模型预测准确性之间的缩放模式。不同序列长度、系统大小和随机性下的数据点都合并成一条曲线，捕捉了模型在面对增加的不确定性时作出正确预测的内在能力。考虑到这条曲线具有评估模型的重要属性，我们将其定义为模型的性能特征曲线。

    The information diffusion prediction on social networks aims to predict future recipients of a message, with practical applications in marketing and social media. While different prediction models all claim to perform well, general frameworks for performance evaluation remain limited. Here, we aim to identify a performance characteristic curve for a model, which captures its performance on tasks of different complexity. We propose a metric based on information entropy to quantify the randomness in diffusion data, then identify a scaling pattern between the randomness and the prediction accuracy of the model. Data points in the patterns by different sequence lengths, system sizes, and randomness all collapse into a single curve, capturing a model's inherent capability of making correct predictions against increased uncertainty. Given that this curve has such important properties that it can be used to evaluate the model, we define it as the performance characteristic curve of the model.
    
[^2]: 通过分离学习解决混淆节点问题

    Clarify Confused Nodes Through Separated Learning. (arXiv:2306.02285v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02285](http://arxiv.org/abs/2306.02285)

    本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。

    

    图神经网络（GNN）在图导向任务中取得了显著的进展。然而，现实世界的图中不可避免地包含一定比例的异质节点，这挑战了经典GNN的同质性假设，并阻碍了其性能。现有研究大多数仍设计了具有异质节点和同质节点间共享权重的通用模型。尽管这些努力中包含了高阶信息和多通道架构，但往往效果不佳。少数研究尝试训练不同节点组的分离学习，但受到了不合适的分离度量和低效率的影响。本文首先提出了一种新的度量指标，称为邻域混淆（NC），以便更可靠地分离节点。我们观察到具有不同NC值的节点组在组内准确度和可视化嵌入上存在一定差异。这为基于邻域混淆的图卷积网络（NC-GCN）铺平了道路。

    Graph neural networks (GNNs) have achieved remarkable advances in graph-oriented tasks. However, real-world graphs invariably contain a certain proportion of heterophilous nodes, challenging the homophily assumption of classical GNNs and hindering their performance. Most existing studies continue to design generic models with shared weights between heterophilous and homophilous nodes. Despite the incorporation of high-order messages or multi-channel architectures, these efforts often fall short. A minority of studies attempt to train different node groups separately but suffer from inappropriate separation metrics and low efficiency. In this paper, we first propose a new metric, termed Neighborhood Confusion (NC), to facilitate a more reliable separation of nodes. We observe that node groups with different levels of NC values exhibit certain differences in intra-group accuracy and visualized embeddings. These pave the way for Neighborhood Confusion-guided Graph Convolutional Network (N
    

