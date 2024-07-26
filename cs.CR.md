# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dr. Jekyll and Mr. Hyde: Two Faces of LLMs](https://arxiv.org/abs/2312.03853) | 本研究通过让ChatGPT和Bard冒充复杂人物角色，绕过了安全机制和专门训练程序，展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。 |
| [^2] | [Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification.](http://arxiv.org/abs/2301.08403) | 本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。 |
| [^3] | [Nonparametric extensions of randomized response for private confidence sets.](http://arxiv.org/abs/2202.08728) | 本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。 |

# 详细

[^1]: LLMs的两面性：Jekyll博士与Hyde先生

    Dr. Jekyll and Mr. Hyde: Two Faces of LLMs

    [https://arxiv.org/abs/2312.03853](https://arxiv.org/abs/2312.03853)

    本研究通过让ChatGPT和Bard冒充复杂人物角色，绕过了安全机制和专门训练程序，展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。

    

    仅仅一年前，我们目睹了大型语言模型（LLMs）的使用增加，尤其是在结合像聊天机器人助手之类的应用时。为了防止这些助手产生不当回应，我们实施了安全机制和专门的训练程序。在这项工作中，我们通过让ChatGPT和Bard（以及在某种程度上是Bing chat）冒充复杂人物角色，绕过了这些措施，这些角色与它们本应成为的真实助手的特征相反。我们首先创造出这些人物角色的复杂传记，然后在同一聊天机器人中使用它们进行新的对话。我们的对话采用角色扮演风格，以获得助手不被允许提供的回应。通过使用人物角色，我们展示了被禁止的回应实际上被提供了，从而有可能获取未经授权、非法或有害的信息。这项工作表明，通过使用对抗性pe

    arXiv:2312.03853v2 Announce Type: replace-cross  Abstract: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial pe
    
[^2]: 通过子序列相似性生成序列：理论及其在无人机识别中的应用

    Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification. (arXiv:2301.08403v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.08403](http://arxiv.org/abs/2301.08403)

    本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。

    

    生成人工合成序列的能力在广泛的应用中至关重要，而深度学习架构和生成框架的最新进展已经极大地促进了这一过程。本文使用一种单次生成模型来采样，通过相似性生成子序列，并证明了子序列相似性对整个序列相似性的影响，给出了相应的界限。我们使用一种一次性生成模型来从单个序列的范围内取样，并生成子序列相似的序列，证明了数据集增强方面的实用性。

    The ability to generate synthetic sequences is crucial for a wide range of applications, and recent advances in deep learning architectures and generative frameworks have greatly facilitated this process. Particularly, unconditional one-shot generative models constitute an attractive line of research that focuses on capturing the internal information of a single image or video to generate samples with similar contents. Since many of those one-shot models are shifting toward efficient non-deep and non-adversarial approaches, we examine the versatility of a one-shot generative model for augmenting whole datasets. In this work, we focus on how similarity at the subsequence level affects similarity at the sequence level, and derive bounds on the optimal transport of real and generated sequences based on that of corresponding subsequences. We use a one-shot generative model to sample from the vicinity of individual sequences and generate subsequence-similar ones and demonstrate the improvem
    
[^3]: 随机响应私有置信集的非参数扩展

    Nonparametric extensions of randomized response for private confidence sets. (arXiv:2202.08728v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.08728](http://arxiv.org/abs/2202.08728)

    本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。

    

    本文提出了一种在局部差分隐私（LDP）约束下执行非参数、非渐进统计推断的方法，用于计算具有均值$\mu^\star$的有界观测$(X_1,\dots,X_n)$的置信区间（CI）和时间均匀置信序列（CS），当只有访问私有数据$(Z_1,\dots,Z_n)$时。为了实现这一点，我们引入了一个非参数的、顺序交互的 Warner 的著名“随机响应”机制的推广，为任意有界随机变量满足 LDP，并提供 CIs 和 CSs，用于访问所得私有化的观测值的均值。例如，我们的结果在固定时间和时间均匀区域都产生了 Hoeffding 不等式的私有模拟。我们将这些 Hoeffding  类型的 CSs 扩展到捕获时间变化（非平稳）的均值，最后说明了如何利用这些方法进行实证。

    This work derives methods for performing nonparametric, nonasymptotic statistical inference for population means under the constraint of local differential privacy (LDP). Given bounded observations $(X_1, \dots, X_n)$ with mean $\mu^\star$ that are privatized into $(Z_1, \dots, Z_n)$, we present confidence intervals (CI) and time-uniform confidence sequences (CS) for $\mu^\star$ when only given access to the privatized data. To achieve this, we introduce a nonparametric and sequentially interactive generalization of Warner's famous ``randomized response'' mechanism, satisfying LDP for arbitrary bounded random variables, and then provide CIs and CSs for their means given access to the resulting privatized observations. For example, our results yield private analogues of Hoeffding's inequality in both fixed-time and time-uniform regimes. We extend these Hoeffding-type CSs to capture time-varying (non-stationary) means, and conclude by illustrating how these methods can be used to conduct
    

