# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information.](http://arxiv.org/abs/2210.00116) | 本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。 |

# 详细

[^1]: 利用变分因果推断和精细关系信息预测细胞响应

    Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information. (arXiv:2210.00116v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.00116](http://arxiv.org/abs/2210.00116)

    本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。

    

    预测细胞在干扰下的响应可能为药物研发和个性化治疗带来重要好处。在本研究中，我们提出了一种新的图形变分贝叶斯因果推断框架，预测细胞在反事实干扰下（即细胞未真实接收的干扰）的基因表达，利用代表生物学知识的基因调控网络（GRN）信息来辅助个性化细胞响应预测。我们还针对数据自适应GRN开发了邻接矩阵更新技术用于图卷积网络的预训练，在模型性能上提供了更多的基因关系洞见。

    Predicting the responses of a cell under perturbations may bring important benefits to drug discovery and personalized therapeutics. In this work, we propose a novel graph variational Bayesian causal inference framework to predict a cell's gene expressions under counterfactual perturbations (perturbations that this cell did not factually receive), leveraging information representing biological knowledge in the form of gene regulatory networks (GRNs) to aid individualized cellular response predictions. Aiming at a data-adaptive GRN, we also developed an adjacency matrix updating technique for graph convolutional networks and used it to refine GRNs during pre-training, which generated more insights on gene relations and enhanced model performance. Additionally, we propose a robust estimator within our framework for the asymptotically efficient estimation of marginal perturbation effect, which is yet to be carried out in previous works. With extensive experiments, we exhibited the advanta
    

