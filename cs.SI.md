# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Large Language Models Detect Misinformation in Scientific News Reporting?](https://arxiv.org/abs/2402.14268) | 大型语言模型探测科学报道中的错误信息的可行性，绕过生成明确标记索赔的步骤，处理现实场景中可能不存在明确标记索赔的挑战。 |
| [^2] | [Lost in Translation -- Multilingual Misinformation and its Evolution.](http://arxiv.org/abs/2310.18089) | 本文通过对超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。结果显示，部分错误信息能够穿越语言障碍，并且在相同语言中更有可能传播。研究还发现错误信息随时间演变并在不同语言间发生突变。 |
| [^3] | [When Does Bottom-up Beat Top-down in Hierarchical Community Detection?.](http://arxiv.org/abs/2306.00833) | 本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。 |

# 详细

[^1]: 大型语言模型能够检测科学新闻报道中的错误信息吗？

    Can Large Language Models Detect Misinformation in Scientific News Reporting?

    [https://arxiv.org/abs/2402.14268](https://arxiv.org/abs/2402.14268)

    大型语言模型探测科学报道中的错误信息的可行性，绕过生成明确标记索赔的步骤，处理现实场景中可能不存在明确标记索赔的挑战。

    

    科学事实经常被在流行媒体中操纵，意图影响公众舆论和行动，正如在COVID-19大流行期间所证实的那样。在科学领域中自动检测错误信息具有挑战性，因为这两种媒体类型的写作风格有着明显不同，并且仍处于萌芽阶段。本文的核心研究问题是是否可以利用大型语言模型(LLMs)来检测科学报道中的错误信息。

    arXiv:2402.14268v1 Announce Type: cross  Abstract: Scientific facts are often spun in the popular press with the intent to influence public opinion and action, as was evidenced during the COVID-19 pandemic. Automatic detection of misinformation in the scientific domain is challenging because of the distinct styles of writing in these two media types and is still in its nascence. Most research on the validity of scientific reporting treats this problem as a claim verification challenge. In doing so, significant expert human effort is required to generate appropriate claims. Our solution bypasses this step and addresses a more real-world scenario where such explicit, labeled claims may not be available. The central research question of this paper is whether it is possible to use large language models (LLMs) to detect misinformation in scientific reporting. To this end, we first present a new labeled dataset SciNews, containing 2.4k scientific news stories drawn from trusted and untrustwo
    
[^2]: 在翻译中迷失-多语言错误信息及其演变

    Lost in Translation -- Multilingual Misinformation and its Evolution. (arXiv:2310.18089v1 [cs.CL])

    [http://arxiv.org/abs/2310.18089](http://arxiv.org/abs/2310.18089)

    本文通过对超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。结果显示，部分错误信息能够穿越语言障碍，并且在相同语言中更有可能传播。研究还发现错误信息随时间演变并在不同语言间发生突变。

    

    在数字时代，误导和虚假信息正在迅速在各种语言和边界间传播，构成了日益增长的威胁。本文通过对95种语言中超过250,000个唯一事实核查的分析，探究了多语言错误信息的普遍性和动态性。首先，我们发现大多数错误信息主张仅被事实核查一次，但11.7%的主张(超过21,000个)被核查多次。运用事实核查作为错误信息传播的代理指标，我们发现33%的重复主张穿越语言障碍，暗示部分错误信息渗透了语言边界。然而，扩散模式表现出较强的同质性，错误信息更有可能在相同语言中传播。为研究主张随时间的演变和跨语言的突变，我们使用多语言句子嵌入来表示事实核查，并对语义相似的主张进行聚类。我们分析了连接组件和最短路径。

    Misinformation and disinformation are growing threats in the digital age, spreading rapidly across languages and borders. This paper investigates the prevalence and dynamics of multilingual misinformation through an analysis of over 250,000 unique fact-checks spanning 95 languages. First, we find that while the majority of misinformation claims are only fact-checked once, 11.7%, corresponding to more than 21,000 claims, are checked multiple times. Using fact-checks as a proxy for the spread of misinformation, we find 33% of repeated claims cross linguistic boundaries, suggesting that some misinformation permeates language barriers. However, spreading patterns exhibit strong homophily, with misinformation more likely to spread within the same language. To study the evolution of claims over time and mutations across languages, we represent fact-checks with multilingual sentence embeddings and cluster semantically similar claims. We analyze the connected components and shortest paths conn
    
[^3]: 自下而上何时击败自上而下进行分层社区检测？

    When Does Bottom-up Beat Top-down in Hierarchical Community Detection?. (arXiv:2306.00833v1 [cs.SI])

    [http://arxiv.org/abs/2306.00833](http://arxiv.org/abs/2306.00833)

    本文研究了使用自下而上算法恢复Hierarchical Stochastic Block Model的树形结构和社区结构的理论保证，并确定了其在中间层次上达到了确切恢复信息理论阈值。

    

    网络的分层聚类是指查找一组社区的树形结构，其中层次结构的较低级别显示更细粒度的社区结构。解决这一问题的算法有两个主要类别：自上而下的算法和自下而上的算法。本文研究了使用自下而上算法恢复分层随机块模型的树形结构和社区结构的理论保证。我们还确定了这种自下而上算法在层次结构的中间层次上达到了确切恢复信息理论阈值。值得注意的是，这些恢复条件相对于现有的自上而下算法的条件来说，限制更少。

    Hierarchical clustering of networks consists in finding a tree of communities, such that lower levels of the hierarchy reveal finer-grained community structures. There are two main classes of algorithms tackling this problem. Divisive ($\textit{top-down}$) algorithms recursively partition the nodes into two communities, until a stopping rule indicates that no further split is needed. In contrast, agglomerative ($\textit{bottom-up}$) algorithms first identify the smallest community structure and then repeatedly merge the communities using a $\textit{linkage}$ method. In this article, we establish theoretical guarantees for the recovery of the hierarchical tree and community structure of a Hierarchical Stochastic Block Model by a bottom-up algorithm. We also establish that this bottom-up algorithm attains the information-theoretic threshold for exact recovery at intermediate levels of the hierarchy. Notably, these recovery conditions are less restrictive compared to those existing for to
    

