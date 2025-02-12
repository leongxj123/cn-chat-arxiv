# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What Changed? Converting Representational Interventions to Natural Language](https://arxiv.org/abs/2402.11355) | 将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。 |
| [^2] | [(Ir)rationality in AI: State of the Art, Research Challenges and Open Questions](https://arxiv.org/abs/2311.17165) | 这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。 |
| [^3] | [FairDP: Certified Fairness with Differential Privacy.](http://arxiv.org/abs/2305.16474) | FairDP是一种同时确保差分隐私和公平性的新型机制，通过独立为不同的个体群体训练模型，在训练过程中逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。相比现有方法，FairDP展示了更好的模型效益、隐私和公平性的权衡。 |

# 详细

[^1]: 改变了什么？将表征干预转化为自然语言

    What Changed? Converting Representational Interventions to Natural Language

    [https://arxiv.org/abs/2402.11355](https://arxiv.org/abs/2402.11355)

    将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。

    

    针对语言模型（LMs）表征空间的干预方法已经被证明是影响模型行为的有效手段。这些方法被用来消除或改变模型表示中的人口统计信息（如性别）的编码，创建一个反事实的表示。然而，由于干预操作在表示空间内，准确理解它修改了哪些特征是一个挑战。我们展示了表征空间的反事实可以转化为自然语言的反事实。我们证明了这种方法使我们能够分析对应于给定表示空间干预的语言变化，并解释用于编码特定概念的特征。此外，由此产生的反事实可以用于减轻分类中的偏见。

    arXiv:2402.11355v1 Announce Type: new  Abstract: Interventions targeting the representation space of language models (LMs) have emerged as effective means to influence model behavior. These methods are employed, for example, to eliminate or alter the encoding of demographic information such as gender within the model's representations, creating a counterfactual representation. However, since the intervention operates within the representation space, understanding precisely which features it modifies poses a challenge. We show that representation-space counterfactuals can be converted into natural language counterfactuals. We demonstrate that this approach enables us to analyze the linguistic alterations corresponding to a given representation-space intervention and to interpret the features utilized for encoding a specific concept. Moreover, the resulting counterfactuals can be used to mitigate bias in classification.
    
[^2]: (非)理性在人工智能中的应用：现状、研究挑战和未解之问

    (Ir)rationality in AI: State of the Art, Research Challenges and Open Questions

    [https://arxiv.org/abs/2311.17165](https://arxiv.org/abs/2311.17165)

    这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。

    

    理性概念在人工智能领域中占据着重要地位。无论是模拟人类推理还是追求有限最优性，我们通常希望使人工智能代理尽可能理性。尽管这个概念在人工智能中非常核心，但对于什么构成理性代理并没有统一的定义。本文调查了人工智能中的理性与非理性，并提出了这个领域的未解问题。在其他领域对理性的理解对其在人工智能中的概念产生了影响，特别是经济学、哲学和心理学方面的研究。着重考虑人工智能代理的行为，我们探讨了在某些情境中非理性行为可能是最优的情况。关于处理非理性代理的方法已经得到了一些发展，包括识别和交互等方面的研究，然而，在这个领域的工作仍然存在一些挑战和问题。

    arXiv:2311.17165v2 Announce Type: replace Abstract: The concept of rationality is central to the field of artificial intelligence. Whether we are seeking to simulate human reasoning, or the goal is to achieve bounded optimality, we generally seek to make artificial agents as rational as possible. Despite the centrality of the concept within AI, there is no unified definition of what constitutes a rational agent. This article provides a survey of rationality and irrationality in artificial intelligence, and sets out the open questions in this area. The understanding of rationality in other fields has influenced its conception within artificial intelligence, in particular work in economics, philosophy and psychology. Focusing on the behaviour of artificial agents, we consider irrational behaviours that can prove to be optimal in certain scenarios. Some methods have been developed to deal with irrational agents, both in terms of identification and interaction, however work in this area re
    
[^3]: FairDP: 具有差分隐私认证的公平性保障

    FairDP: Certified Fairness with Differential Privacy. (arXiv:2305.16474v1 [cs.LG])

    [http://arxiv.org/abs/2305.16474](http://arxiv.org/abs/2305.16474)

    FairDP是一种同时确保差分隐私和公平性的新型机制，通过独立为不同的个体群体训练模型，在训练过程中逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。相比现有方法，FairDP展示了更好的模型效益、隐私和公平性的权衡。

    

    本文介绍了一种名为FairDP的新型机制，旨在同时确保差分隐私(DP)和公平性。FairDP通过独立为不同的个体群体训练模型，在使用组特定的剪裁项来评估和限制DP的差异影响的同时操作。在训练过程中，该机制逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。广泛的理论和实证分析验证了FairDP的功效，与现有方法相比，展示了更好的模型效益、隐私和公平性的权衡。

    This paper introduces FairDP, a novel mechanism designed to simultaneously ensure differential privacy (DP) and fairness. FairDP operates by independently training models for distinct individual groups, using group-specific clipping terms to assess and bound the disparate impacts of DP. Throughout the training process, the mechanism progressively integrates knowledge from group models to formulate a comprehensive model that balances privacy, utility, and fairness in downstream tasks. Extensive theoretical and empirical analyses validate the efficacy of FairDP, demonstrating improved trade-offs between model utility, privacy, and fairness compared with existing methods.
    

