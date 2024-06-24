# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentivizing High-Quality Content in Online Recommender Systems.](http://arxiv.org/abs/2306.07479) | 本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。 |

# 详细

[^1]: 在在线推荐系统中激励高质量内容

    Incentivizing High-Quality Content in Online Recommender Systems. (arXiv:2306.07479v1 [cs.GT])

    [http://arxiv.org/abs/2306.07479](http://arxiv.org/abs/2306.07479)

    本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。

    

    对于像TikTok和YouTube这样的内容推荐系统，平台的决策算法塑造了内容生产者的激励，包括生产者在内容质量上投入多少努力。许多平台采用在线学习，这会产生跨时间的激励，因为今天生产的内容会影响未来内容的推荐。在本文中，我们研究了在线学习产生的激励，分析了在纳什均衡下生产的内容质量。我们发现，像Hedge和EXP3这样的经典在线学习算法会激励生产者创建低质量的内容。特别地，内容质量在学习率方面有上限，并且随着典型学习率进展而趋近于零。在这一负面结果的基础上，我们设计了一种不同的学习算法——基于惩罚创建低质量内容的生产者——正确激励生产者创建高质量内容。我们的算法依赖于新颖的策略性赌博机问题，并克服了在组合设置中应用对抗性技术的挑战。在模拟和真实数据的实验中，我们的算法成功地激励生产者创建高质量内容。

    For content recommender systems such as TikTok and YouTube, the platform's decision algorithm shapes the incentives of content producers, including how much effort the content producers invest in the quality of their content. Many platforms employ online learning, which creates intertemporal incentives, since content produced today affects recommendations of future content. In this paper, we study the incentives arising from online learning, analyzing the quality of content produced at a Nash equilibrium. We show that classical online learning algorithms, such as Hedge and EXP3, unfortunately incentivize producers to create low-quality content. In particular, the quality of content is upper bounded in terms of the learning rate and approaches zero for typical learning rate schedules. Motivated by this negative result, we design a different learning algorithm -- based on punishing producers who create low-quality content -- that correctly incentivizes producers to create high-quality co
    

