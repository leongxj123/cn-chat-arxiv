# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning](https://arxiv.org/abs/2402.11933) | SLADE通过自监督学习在边缘流中迅速检测动态异常，无需依赖标签，主要通过观察节点交互模式的偏差来检测节点状态转变。 |
| [^2] | [Predicting the structure of dynamic graphs.](http://arxiv.org/abs/2401.04280) | 本文提出了一种预测动态图结构的方法，利用时间序列方法预测未来时间点的节点度，并结合通量平衡分析方法获得未来图的结构，评估了该方法在合成和真实数据集上的实用性和适用性。 |
| [^3] | [Brand Network Booster: A New System for Improving Brand Connectivity.](http://arxiv.org/abs/2309.16228) | 本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。 |

# 详细

[^1]: SLADE：通过自监督学习在边缘流中检测动态异常

    SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning

    [https://arxiv.org/abs/2402.11933](https://arxiv.org/abs/2402.11933)

    SLADE通过自监督学习在边缘流中迅速检测动态异常，无需依赖标签，主要通过观察节点交互模式的偏差来检测节点状态转变。

    

    为了检测真实世界图中的异常，如社交、电子邮件和金融网络，已经开发了各种方法。在大多数真实世界图随时间增长，自然地表示为边缘流的情况下，我们的目标是：(a)在异常发生时即时检测异常，(b)适应动态变化的状态，(c)处理动态异常标签的稀缺性。在本文中，我们提出了SLADE（边缘流异常检测的自监督学习），用于在边缘流中快速检测动态异常，而不依赖于标签。SLADE通过观察节点在时间上相互作用模式的偏差来检测节点进入异常状态的转变。为此，它训练一个深度神经网络执行两个自监督任务：(a)最小化节点表示中的漂移，(b)从短期生成长期交互模式。

    arXiv:2402.11933v1 Announce Type: new  Abstract: To detect anomalies in real-world graphs, such as social, email, and financial networks, various approaches have been developed. While they typically assume static input graphs, most real-world graphs grow over time, naturally represented as edge streams. In this context, we aim to achieve three goals: (a) instantly detecting anomalies as they occur, (b) adapting to dynamically changing states, and (c) handling the scarcity of dynamic anomaly labels. In this paper, we propose SLADE (Self-supervised Learning for Anomaly Detection in Edge Streams) for rapid detection of dynamic anomalies in edge streams, without relying on labels. SLADE detects the shifts of nodes into abnormal states by observing deviations in their interaction patterns over time. To this end, it trains a deep neural network to perform two self-supervised tasks: (a) minimizing drift in node representations and (b) generating long-term interaction patterns from short-term 
    
[^2]: 预测动态图的结构

    Predicting the structure of dynamic graphs. (arXiv:2401.04280v1 [cs.LG])

    [http://arxiv.org/abs/2401.04280](http://arxiv.org/abs/2401.04280)

    本文提出了一种预测动态图结构的方法，利用时间序列方法预测未来时间点的节点度，并结合通量平衡分析方法获得未来图的结构，评估了该方法在合成和真实数据集上的实用性和适用性。

    

    动态图嵌入、归纳和增量学习有助于预测任务，如节点分类和链接预测。然而，从图的时间序列中预测未来时间步的图结构，允许有新节点，并没有受到太多关注。在本文中，我们提出了一种这样的方法。我们使用时间序列方法预测未来时间点的节点度，并将其与通量平衡分析（一种在生物化学中使用的线性规划方法）结合起来，以获得未来图的结构。此外，我们探索了不同参数值的预测图分布。我们使用合成和真实数据集评估了该方法，并展示了其实用性和适用性。

    Dynamic graph embeddings, inductive and incremental learning facilitate predictive tasks such as node classification and link prediction. However, predicting the structure of a graph at a future time step from a time series of graphs, allowing for new nodes has not gained much attention. In this paper, we present such an approach. We use time series methods to predict the node degree at future time points and combine it with flux balance analysis -- a linear programming method used in biochemistry -- to obtain the structure of future graphs. Furthermore, we explore the predictive graph distribution for different parameter values. We evaluate this method using synthetic and real datasets and demonstrate its utility and applicability.
    
[^3]: 品牌网络增强器：提升品牌连接性的新系统

    Brand Network Booster: A New System for Improving Brand Connectivity. (arXiv:2309.16228v1 [cs.SI])

    [http://arxiv.org/abs/2309.16228](http://arxiv.org/abs/2309.16228)

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。

    

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。在网络分析方面，我们通过解决扩展版的最大连接度改进问题来实现这一目标，其中包括考虑敌对节点、约束预算和加权网络的可能性 - 通过添加链接或增加现有连接的权重来实现连接性的改进。我们结合两个案例研究来展示这个新系统，并讨论其性能。我们的工具和方法对于网络学者和支持市场营销和传播管理者的战略决策过程都很有用。

    This paper presents a new decision support system offered for an in-depth analysis of semantic networks, which can provide insights for a better exploration of a brand's image and the improvement of its connectivity. In terms of network analysis, we show that this goal is achieved by solving an extended version of the Maximum Betweenness Improvement problem, which includes the possibility of considering adversarial nodes, constrained budgets, and weighted networks - where connectivity improvement can be obtained by adding links or increasing the weight of existing connections. We present this new system together with two case studies, also discussing its performance. Our tool and approach are useful both for network scholars and for supporting the strategic decision-making processes of marketing and communication managers.
    

