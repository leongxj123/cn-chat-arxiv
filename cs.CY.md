# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs](https://arxiv.org/abs/2403.09724) | ClaimVer是一个人为中心的框架，通过知识图谱实现可解释的声明级验证和证据归因，致力于提高用户对文本验证方法的信任并强调细粒度证据的重要性。 |
| [^2] | [Content Conditional Debiasing for Fair Text Embedding](https://arxiv.org/abs/2402.14208) | 通过在内容条件下确保敏感属性与文本嵌入之间的条件独立性，我们提出了一种可以改善公平性的新方法，在保持效用的同时，解决了缺乏适当训练数据的问题。 |
| [^3] | [Causal Fair Machine Learning via Rank-Preserving Interventional Distributions.](http://arxiv.org/abs/2307.12797) | 通过保持排序的干预分布，我们提出了一种因果公平的机器学习方法，通过在一个理想的世界中消除受保护属性对目标的因果影响来减少不公平。 |

# 详细

[^1]: ClaimVer：通过知识图谱实现可解释的声明级验证和证据归因

    ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs

    [https://arxiv.org/abs/2403.09724](https://arxiv.org/abs/2403.09724)

    ClaimVer是一个人为中心的框架，通过知识图谱实现可解释的声明级验证和证据归因，致力于提高用户对文本验证方法的信任并强调细粒度证据的重要性。

    

    在广泛传播的信息误导和社交媒体以及人工智能生成的文本的激增中，验证和信任所遇到的信息变得日益困难。许多事实核查方法和工具已被开发，但它们往往缺乏适当的可解释性或细粒度，无法在各种情境中发挥作用。一种易于使用、可访问且能够执行细粒度证据归因的文本验证方法变得至关重要。更重要的是，建立用户对这种方法的信任需要呈现每个预测背后的理由，因为研究表明这显著影响人们对自动化系统的信任。将用户关注重点放在具体的问题内容上，而不是提供简单的笼统标签也非常重要。在本文中，我们提出了$\textit{ClaimVer，一个以人为中心的框架}$，旨在满足用户的信息需求。

    arXiv:2403.09724v1 Announce Type: new  Abstract: In the midst of widespread misinformation and disinformation through social media and the proliferation of AI-generated texts, it has become increasingly difficult for people to validate and trust information they encounter. Many fact-checking approaches and tools have been developed, but they often lack appropriate explainability or granularity to be useful in various contexts. A text validation method that is easy to use, accessible, and can perform fine-grained evidence attribution has become crucial. More importantly, building user trust in such a method requires presenting the rationale behind each prediction, as research shows this significantly influences people's belief in automated systems. It is also paramount to localize and bring users' attention to the specific problematic content, instead of providing simple blanket labels. In this paper, we present $\textit{ClaimVer, a human-centric framework}$ tailored to meet users' info
    
[^2]: 面向公平文本嵌入的内容条件去偏方法

    Content Conditional Debiasing for Fair Text Embedding

    [https://arxiv.org/abs/2402.14208](https://arxiv.org/abs/2402.14208)

    通过在内容条件下确保敏感属性与文本嵌入之间的条件独立性，我们提出了一种可以改善公平性的新方法，在保持效用的同时，解决了缺乏适当训练数据的问题。

    

    在自然语言处理（NLP）中，减轻机器学习模型中的偏见引起了越来越多的关注。然而，只有少数研究集中在公平的文本嵌入上，这对实际应用至关重要且具有挑战性。本文提出了一种学习公平文本嵌入的新方法。我们通过确保在内容条件下敏感属性与文本嵌入之间的条件独立性来实现公平性，同时保持效用权衡。具体来说，我们强制要求具有不同敏感属性但相同内容的文本的嵌入与其对应中立文本的嵌入保持相同的距离。此外，我们通过使用大型语言模型（LLMs）将文本增强为不同的敏感组，来解决缺乏适当训练数据的问题。我们广泛的评估表明，我们的方法有效地提高了公平性同时保持了嵌入的效用。

    arXiv:2402.14208v1 Announce Type: cross  Abstract: Mitigating biases in machine learning models has gained increasing attention in Natural Language Processing (NLP). Yet, only a few studies focus on fair text embeddings, which are crucial yet challenging for real-world applications. In this paper, we propose a novel method for learning fair text embeddings. We achieve fairness while maintaining utility trade-off by ensuring conditional independence between sensitive attributes and text embeddings conditioned on the content. Specifically, we enforce that embeddings of texts with different sensitive attributes but identical content maintain the same distance toward the embedding of their corresponding neutral text. Furthermore, we address the issue of lacking proper training data by using Large Language Models (LLMs) to augment texts into different sensitive groups. Our extensive evaluations demonstrate that our approach effectively improves fairness while preserving the utility of embed
    
[^3]: 通过保持排序的干预分布实现因果公平的机器学习

    Causal Fair Machine Learning via Rank-Preserving Interventional Distributions. (arXiv:2307.12797v1 [cs.LG])

    [http://arxiv.org/abs/2307.12797](http://arxiv.org/abs/2307.12797)

    通过保持排序的干预分布，我们提出了一种因果公平的机器学习方法，通过在一个理想的世界中消除受保护属性对目标的因果影响来减少不公平。

    

    如果相同的个体得到相同的对待，而不同的个体得到不同的对待，那么一个决策被定义为公平的。根据这个定义，在设计机器学习模型以减少自动决策系统中的不公平时，必须引入因果思考来引入受保护属性。根据最近的提议，我们将个体定义为在一个假设的、理想的（FiND）世界中是规范上相等的，这个世界中受保护属性对目标没有（直接或间接）的因果影响。我们提出保持排序的干预分布来定义这个FiND世界的估计目标，并提出了一个估计方法。通过模拟和实证数据的验证，我们提供了对方法和生成模型的评价标准。通过这些，我们展示了我们的干预方法有效地识别出最受歧视的个体并减少不公平。

    A decision can be defined as fair if equal individuals are treated equally and unequals unequally. Adopting this definition, the task of designing machine learning models that mitigate unfairness in automated decision-making systems must include causal thinking when introducing protected attributes. Following a recent proposal, we define individuals as being normatively equal if they are equal in a fictitious, normatively desired (FiND) world, where the protected attribute has no (direct or indirect) causal effect on the target. We propose rank-preserving interventional distributions to define an estimand of this FiND world and a warping method for estimation. Evaluation criteria for both the method and resulting model are presented and validated through simulations and empirical data. With this, we show that our warping approach effectively identifies the most discriminated individuals and mitigates unfairness.
    

