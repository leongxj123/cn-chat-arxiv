# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting Link Prediction: A Data Perspective](https://arxiv.org/abs/2310.00793) | 本文通过从数据的视角出发，重新审视了链接预测的原则，并发现了局部结构接近性、全局结构接近性和特征接近性三个因素之间的关系。同时，发现了全局结构接近性只在局部结构接近性不足时显示出有效性，以及特征和结构接近性之间的不兼容性。这些发现为链接预测提供了新的思路，启发了GNN4LP的设计。 |
| [^2] | [Using Causality-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs.](http://arxiv.org/abs/2310.15865) | 本研究提出了一种使用因果感知图神经网络预测动态图中的时间中心性的方法，并在不同领域的13个时间图上进行了实验验证，结果显示该方法显著改善了介数和接近度中心性的预测能力。 |

# 详细

[^1]: 重新审视链接预测: 一个数据的视角

    Revisiting Link Prediction: A Data Perspective

    [https://arxiv.org/abs/2310.00793](https://arxiv.org/abs/2310.00793)

    本文通过从数据的视角出发，重新审视了链接预测的原则，并发现了局部结构接近性、全局结构接近性和特征接近性三个因素之间的关系。同时，发现了全局结构接近性只在局部结构接近性不足时显示出有效性，以及特征和结构接近性之间的不兼容性。这些发现为链接预测提供了新的思路，启发了GNN4LP的设计。

    

    链接预测是一项基于图的基本任务，在各种应用中已被证明是不可或缺的，例如朋友推荐、蛋白质分析和药物互作预测。然而，由于数据集涵盖了多个领域，它们可能具有不同的链接形成机制。现有文献中的证据强调了一个普遍适用于所有数据集的最佳算法的缺失。在本文中，我们尝试从数据中心的视角探索链接预测的原则，跨越不同数据集。我们确定了三个对链接预测至关重要的基本因素:局部结构接近性、全局结构接近性和特征接近性。然后，我们揭示了这些因素之间的关系，其中 (i)只有在局部结构接近性不足的情况下，全局结构接近性才显示出有效性。 (ii)特征和结构接近性之间存在不兼容性。这种不兼容性导致了链接预测的图神经网络 (GNN4LP) 持续地

    Link prediction, a fundamental task on graphs, has proven indispensable in various applications, e.g., friend recommendation, protein analysis, and drug interaction prediction. However, since datasets span a multitude of domains, they could have distinct underlying mechanisms of link formation. Evidence in existing literature underscores the absence of a universally best algorithm suitable for all datasets. In this paper, we endeavor to explore principles of link prediction across diverse datasets from a data-centric perspective. We recognize three fundamental factors critical to link prediction: local structural proximity, global structural proximity, and feature proximity. We then unearth relationships among those factors where (i) global structural proximity only shows effectiveness when local structural proximity is deficient. (ii) The incompatibility can be found between feature and structural proximity. Such incompatibility leads to GNNs for Link Prediction (GNN4LP) consistently 
    
[^2]: 使用因果感知图神经网络在动态图中预测时间中心性

    Using Causality-Aware Graph Neural Networks to Predict Temporal Centralities in Dynamic Graphs. (arXiv:2310.15865v1 [cs.LG])

    [http://arxiv.org/abs/2310.15865](http://arxiv.org/abs/2310.15865)

    本研究提出了一种使用因果感知图神经网络预测动态图中的时间中心性的方法，并在不同领域的13个时间图上进行了实验验证，结果显示该方法显著改善了介数和接近度中心性的预测能力。

    

    节点中心性在网络科学、社交网络分析和推荐系统中起着重要作用。在时间数据中，静态基于路径的中心性如接近度或介数可能会对节点在时间图中的真实重要性产生误导。为了解决这个问题，已经定义了基于节点对之间最短时间路径的时间一般化介数和接近度。然而，这些一般化的一个主要问题是计算这样的路径的计算成本较高。为了解决这个问题，我们研究了De Bruijn图神经网络(DBGNN)，一种因果感知的图神经网络架构，在时间序列数据中预测基于路径的时间中心性。我们在13个生物和社交系统的时间图中实验评估了我们的方法，并显示它相比静态图卷积方法显著改善了介数和接近度中心性的预测能力。

    Node centralities play a pivotal role in network science, social network analysis, and recommender systems. In temporal data, static path-based centralities like closeness or betweenness can give misleading results about the true importance of nodes in a temporal graph. To address this issue, temporal generalizations of betweenness and closeness have been defined that are based on the shortest time-respecting paths between pairs of nodes. However, a major issue of those generalizations is that the calculation of such paths is computationally expensive. Addressing this issue, we study the application of De Bruijn Graph Neural Networks (DBGNN), a causality-aware graph neural network architecture, to predict temporal path-based centralities in time series data. We experimentally evaluate our approach in 13 temporal graphs from biological and social systems and show that it considerably improves the prediction of both betweenness and closeness centrality compared to a static Graph Convolut
    

