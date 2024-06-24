# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Impact of Decentralized Learning on Player Utilities in Stackelberg Games](https://arxiv.org/abs/2403.00188) | 研究了分散学习对斯塔克尔贝格博弈中玩家效用的影响，提出了一种放松遗憾基准来更好捕捉系统特征，并开发了实现近乎最优遗憾的算法。 |
| [^2] | [One-Shot Strategic Classification Under Unknown Costs](https://arxiv.org/abs/2311.02761) | 本研究首次研究了在未知响应下一次性策略分类的情景，针对用户成本函数不确定性，提出解决方案并将任务定义为极小-极大问题。 |
| [^3] | [Incentivizing High-Quality Content in Online Recommender Systems.](http://arxiv.org/abs/2306.07479) | 本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。 |

# 详细

[^1]: 分散学习对斯塔克尔贝格博弈中玩家效用的影响

    Impact of Decentralized Learning on Player Utilities in Stackelberg Games

    [https://arxiv.org/abs/2403.00188](https://arxiv.org/abs/2403.00188)

    研究了分散学习对斯塔克尔贝格博弈中玩家效用的影响，提出了一种放松遗憾基准来更好捕捉系统特征，并开发了实现近乎最优遗憾的算法。

    

    当学习代理（如推荐系统或聊天机器人）在现实世界中部署时，通常会随时间反复与另一个学习代理（如用户）交互。在许多这样的双代理系统中，每个代理单独学习，而两个代理的奖励并不完全一致。为了更好地理解这类情况，我们研究了两代理系统的学习动态以及对每个代理目标的影响。我们将这些系统建模为具有分散学习的斯塔克尔贝格博弈，并展示标准遗憾基准（如斯塔克尔贝格均衡回报）导致至少有一名玩家出现最坏情况的线性遗憾。为了更好地捕捉这些系统，我们构建了一种对代理的学习误差容忍的放松遗憾基准。我们展示了标准学习算法未能提供次线性遗憾，并开发了算法，实现了对于双方玩家而言与理想$O(T^{2/3})$遗憾接近最优。

    arXiv:2403.00188v1 Announce Type: new  Abstract: When deployed in the world, a learning agent such as a recommender system or a chatbot often repeatedly interacts with another learning agent (such as a user) over time. In many such two-agent systems, each agent learns separately and the rewards of the two agents are not perfectly aligned. To better understand such cases, we examine the learning dynamics of the two-agent system and the implications for each agent's objective. We model these systems as Stackelberg games with decentralized learning and show that standard regret benchmarks (such as Stackelberg equilibrium payoffs) result in worst-case linear regret for at least one player. To better capture these systems, we construct a relaxed regret benchmark that is tolerant to small learning errors by agents. We show that standard learning algorithms fail to provide sublinear regret, and we develop algorithms to achieve near-optimal $O(T^{2/3})$ regret for both players with respect to 
    
[^2]: 一次性策略分类在未知成本下的研究

    One-Shot Strategic Classification Under Unknown Costs

    [https://arxiv.org/abs/2311.02761](https://arxiv.org/abs/2311.02761)

    本研究首次研究了在未知响应下一次性策略分类的情景，针对用户成本函数不确定性，提出解决方案并将任务定义为极小-极大问题。

    

    策略分类的目标是学习对策略输入操纵具有鲁棒性的决策规则。之前的研究假设这些响应是已知的；而最近的一些研究处理未知响应，但它们专门研究重复模型部署的在线设置。然而，在许多领域，特别是在公共政策中，一个常见的激励用例中，多次部署是不可行的，甚至一个糟糕的轮次都是不可接受的。为了填补这一空白，我们首次引入了在未知响应下的一次性策略分类的正式研究，这需要在一次性选择一个分类器。着重关注用户成本函数中的不确定性，我们首先证明对于一类广泛的成本，即使对真实成本的小误差也可能在最坏情况下导致准确性降至极低水平。鉴于此，我们将任务框定为极小-极大问题，目标是识别

    arXiv:2311.02761v2 Announce Type: replace  Abstract: The goal of strategic classification is to learn decision rules which are robust to strategic input manipulation. Earlier works assume that these responses are known; while some recent works handle unknown responses, they exclusively study online settings with repeated model deployments. But there are many domains$\unicode{x2014}$particularly in public policy, a common motivating use case$\unicode{x2014}$where multiple deployments are infeasible, or where even one bad round is unacceptable. To address this gap, we initiate the formal study of one-shot strategic classification under unknown responses, which requires committing to a single classifier once. Focusing on uncertainty in the users' cost function, we begin by proving that for a broad class of costs, even a small mis-estimation of the true cost can entail trivial accuracy in the worst case. In light of this, we frame the task as a minimax problem, with the goal of identifying
    
[^3]: 在在线推荐系统中激励高质量内容

    Incentivizing High-Quality Content in Online Recommender Systems. (arXiv:2306.07479v1 [cs.GT])

    [http://arxiv.org/abs/2306.07479](http://arxiv.org/abs/2306.07479)

    本文研究了在线推荐系统中激励高质量内容的算法问题，经典的在线学习算法会激励生产者创建低质量的内容，但本文提出的一种算法通过惩罚低质量内容的创建者，成功地激励了生产者创造高质量的内容。

    

    对于像TikTok和YouTube这样的内容推荐系统，平台的决策算法塑造了内容生产者的激励，包括生产者在内容质量上投入多少努力。许多平台采用在线学习，这会产生跨时间的激励，因为今天生产的内容会影响未来内容的推荐。在本文中，我们研究了在线学习产生的激励，分析了在纳什均衡下生产的内容质量。我们发现，像Hedge和EXP3这样的经典在线学习算法会激励生产者创建低质量的内容。特别地，内容质量在学习率方面有上限，并且随着典型学习率进展而趋近于零。在这一负面结果的基础上，我们设计了一种不同的学习算法——基于惩罚创建低质量内容的生产者——正确激励生产者创建高质量内容。我们的算法依赖于新颖的策略性赌博机问题，并克服了在组合设置中应用对抗性技术的挑战。在模拟和真实数据的实验中，我们的算法成功地激励生产者创建高质量内容。

    For content recommender systems such as TikTok and YouTube, the platform's decision algorithm shapes the incentives of content producers, including how much effort the content producers invest in the quality of their content. Many platforms employ online learning, which creates intertemporal incentives, since content produced today affects recommendations of future content. In this paper, we study the incentives arising from online learning, analyzing the quality of content produced at a Nash equilibrium. We show that classical online learning algorithms, such as Hedge and EXP3, unfortunately incentivize producers to create low-quality content. In particular, the quality of content is upper bounded in terms of the learning rate and approaches zero for typical learning rate schedules. Motivated by this negative result, we design a different learning algorithm -- based on punishing producers who create low-quality content -- that correctly incentivizes producers to create high-quality co
    

