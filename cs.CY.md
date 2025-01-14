# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GFairHint: Improving Individual Fairness for Graph Neural Networks via Fairness Hint.](http://arxiv.org/abs/2305.15622) | GFairHint提出了一种新方法，通过辅助链接预测任务学习公平表示，并将其与原始图嵌入连接以增强图神经网络的个体公平性，同时不牺牲性能表现。 |
| [^2] | [ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review.](http://arxiv.org/abs/2305.03123) | 本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。 |

# 详细

[^1]: GFairHint：通过公平性提示提高图神经网络的个体公平性

    GFairHint: Improving Individual Fairness for Graph Neural Networks via Fairness Hint. (arXiv:2305.15622v1 [cs.LG])

    [http://arxiv.org/abs/2305.15622](http://arxiv.org/abs/2305.15622)

    GFairHint提出了一种新方法，通过辅助链接预测任务学习公平表示，并将其与原始图嵌入连接以增强图神经网络的个体公平性，同时不牺牲性能表现。

    

    鉴于机器学习中公平性问题日益受到关注以及图神经网络（GNN）在图数据学习上的卓越表现，GNN中的算法公平性受到了广泛关注。虽然许多现有的研究在群体层面上改善了公平性，但只有少数工作促进了个体公平性，这使得相似的个体具有相似的结果。促进个体公平性的理想框架应该（1）在公平性和性能之间平衡，（2）适应两种常用的个体相似性度量（从外部注释和从输入特征计算），（3）横跨各种GNN模型进行推广，（4）具有高效的计算能力。不幸的是，之前的工作都没有实现所有的理想条件。在这项工作中，我们提出了一种新的方法GFairHint，该方法通过辅助链接预测任务学习公平表示，然后将其与原始图嵌入连接以增强个体公平性。实验结果表明，GFairHint在不牺牲太多性能的情况下提高了个体公平性，并且优于目前的最新方法。

    Given the growing concerns about fairness in machine learning and the impressive performance of Graph Neural Networks (GNNs) on graph data learning, algorithmic fairness in GNNs has attracted significant attention. While many existing studies improve fairness at the group level, only a few works promote individual fairness, which renders similar outcomes for similar individuals. A desirable framework that promotes individual fairness should (1) balance between fairness and performance, (2) accommodate two commonly-used individual similarity measures (externally annotated and computed from input features), (3) generalize across various GNN models, and (4) be computationally efficient. Unfortunately, none of the prior work achieves all the desirables. In this work, we propose a novel method, GFairHint, which promotes individual fairness in GNNs and achieves all aforementioned desirables. GFairHint learns fairness representations through an auxiliary link prediction task, and then concate
    
[^2]: ChatGPT 需要进行SPADE（可持续性、隐私、数字鸿沟和伦理）评估：一项综述。

    ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review. (arXiv:2305.03123v1 [cs.CY])

    [http://arxiv.org/abs/2305.03123](http://arxiv.org/abs/2305.03123)

    本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。

    

    ChatGPT是另一个大型语言模型（LLM），由于其性能和有效的对话能力，在研究和工业界中得到了巨大的关注。最近，许多研究已经发表，以展示ChatGPT和其他LLMs的有效性、效率、集成和情感。相反，本研究关注的是大多数被忽视的重要方面，即可持续性、隐私、数字鸿沟和伦理，并建议不仅仅是ChatGPT，而是在对话机器人类别中的每一个后续入口都应该进行SPADE评估。本文详细讨论了关于ChatGPT的问题和关注点与上述特征一致。我们通过一些初步的数据收集和可视化以及假设的事实来支持我们的假设。我们还为每个问题提出了缓解和建议。此外，我们还提供了一些未来方向和开放问题的探讨。

    ChatGPT is another large language model (LLM) inline but due to its performance and ability to converse effectively, it has gained a huge popularity amongst research as well as industrial community. Recently, many studies have been published to show the effectiveness, efficiency, integration, and sentiments of chatGPT and other LLMs. In contrast, this study focuses on the important aspects that are mostly overlooked, i.e. sustainability, privacy, digital divide, and ethics and suggests that not only chatGPT but every subsequent entry in the category of conversational bots should undergo Sustainability, PrivAcy, Digital divide, and Ethics (SPADE) evaluation. This paper discusses in detail about the issues and concerns raised over chatGPT in line with aforementioned characteristics. We support our hypothesis by some preliminary data collection and visualizations along with hypothesized facts. We also suggest mitigations and recommendations for each of the concerns. Furthermore, we also s
    

