# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lost in Translation -- Multilingual Misinformation and its Evolution.](http://arxiv.org/abs/2310.18089) | 本文通过对超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。结果显示，部分错误信息能够穿越语言障碍，并且在相同语言中更有可能传播。研究还发现错误信息随时间演变并在不同语言间发生突变。 |
| [^2] | [Fairness in Streaming Submodular Maximization over a Matroid Constraint.](http://arxiv.org/abs/2305.15118) | 这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。 |

# 详细

[^1]: 在翻译中迷失-多语言错误信息及其演变

    Lost in Translation -- Multilingual Misinformation and its Evolution. (arXiv:2310.18089v1 [cs.CL])

    [http://arxiv.org/abs/2310.18089](http://arxiv.org/abs/2310.18089)

    本文通过对超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。结果显示，部分错误信息能够穿越语言障碍，并且在相同语言中更有可能传播。研究还发现错误信息随时间演变并在不同语言间发生突变。

    

    在数字时代，误导和虚假信息正在迅速在各种语言和边界间传播，构成了日益增长的威胁。本文通过对95种语言中超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。首先，我们发现大多数错误信息主张仅被事实核查一次，但11.7%的主张(超过21,000个)被核查多次。运用事实核查作为错误信息传播的代理指标，我们发现33%的重复主张穿越语言障碍，暗示部分错误信息渗透了语言边界。然而，扩散模式表现出较强的同质性，错误信息更有可能在相同语言中传播。为研究主张随时间的演变和跨语言的突变，我们使用多语言句子嵌入来表示事实核查，并对语义相似的主张进行聚类。我们分析了连接组件和最短路径。

    Misinformation and disinformation are growing threats in the digital age, spreading rapidly across languages and borders. This paper investigates the prevalence and dynamics of multilingual misinformation through an analysis of over 250,000 unique fact-checks spanning 95 languages. First, we find that while the majority of misinformation claims are only fact-checked once, 11.7%, corresponding to more than 21,000 claims, are checked multiple times. Using fact-checks as a proxy for the spread of misinformation, we find 33% of repeated claims cross linguistic boundaries, suggesting that some misinformation permeates language barriers. However, spreading patterns exhibit strong homophily, with misinformation more likely to spread within the same language. To study the evolution of claims over time and mutations across languages, we represent fact-checks with multilingual sentence embeddings and cluster semantically similar claims. We analyze the connected components and shortest paths conn
    
[^2]: 在一个Matroid约束下流式子模最大化中的公平性

    Fairness in Streaming Submodular Maximization over a Matroid Constraint. (arXiv:2305.15118v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15118](http://arxiv.org/abs/2305.15118)

    这篇论文研究了在一个Matroid约束下流式子模最大化中的公平性问题，提供了流式算法和不可能的结果来权衡效率、质量和公平性，并在现实世界应用中进行了实证验证。

    

    流式子模最大化是从一个大规模数据集中选择一个代表性子集的自然模型。如果数据点具有敏感属性，如性别或种族，强制公平性以避免偏见和歧视变得重要。这引起了对开发公平机器学习算法的极大兴趣。最近，这样的算法已经被开发用于基于基数约束的单调子模最大化。在本文中，我们研究了这个问题的自然推广到一个Matroid约束。我们提供了流式算法以及不可能的结果，这些结果在效率、质量和公平性之间提供了权衡。我们在一系列知名的现实世界应用中对我们的发现进行了经验证实：基于示例的聚类、电影推荐和社交网络中的最大覆盖。

    Streaming submodular maximization is a natural model for the task of selecting a representative subset from a large-scale dataset. If datapoints have sensitive attributes such as gender or race, it becomes important to enforce fairness to avoid bias and discrimination. This has spurred significant interest in developing fair machine learning algorithms. Recently, such algorithms have been developed for monotone submodular maximization under a cardinality constraint.  In this paper, we study the natural generalization of this problem to a matroid constraint. We give streaming algorithms as well as impossibility results that provide trade-offs between efficiency, quality and fairness. We validate our findings empirically on a range of well-known real-world applications: exemplar-based clustering, movie recommendation, and maximum coverage in social networks.
    

