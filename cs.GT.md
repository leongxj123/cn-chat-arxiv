# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning in Mean Field Games: A Survey](https://arxiv.org/abs/2205.12944) | 强化学习和均场博弈的结合有望在很大规模上解决游戏的均衡和社会最优问题。 |
| [^2] | [Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models.](http://arxiv.org/abs/2310.00322) | 本文提出了红队游戏（RTG）框架，利用博弈论分析了红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。同时引入了游戏化红队求解器（GRTS）来提供自动化的红队技术。 |
| [^3] | [Bayesian Opponent Modeling in Multiplayer Imperfect-Information Games.](http://arxiv.org/abs/2212.06027) | 该论文提出了一种针对多人不完全信息博弈的贝叶斯对手建模方法，在三人 Kuhn poker 中应用这种方法可以明显超过所有的代理商，包括准确的纳什均衡策略。 |

# 详细

[^1]: 在均场博弈中的学习：一项调查

    Learning in Mean Field Games: A Survey

    [https://arxiv.org/abs/2205.12944](https://arxiv.org/abs/2205.12944)

    强化学习和均场博弈的结合有望在很大规模上解决游戏的均衡和社会最优问题。

    

    非合作和合作游戏在拥有大量玩家时有许多应用，但随着玩家数量的增加，通常变得难以解决。均场博弈(Mean Field Games, MFGs)由Lasry和Lions以及Huang，Caines和Malham\'e引入，依靠均场近似允许玩家数量增长到无穷大。传统解决这些游戏的方法通常依赖于解决带有对模型的完全了解的偏微分方程或随机微分方程。最近，强化学习(Reinforcement Learning, RL)出现在解决规模复杂问题上表现出了很大的潜力。RL和MFGs的结合有望解决在人口规模和环境复杂性方面非常庞大的游戏。在这项调查中，我们回顾了最近迅速增长的关于RL方法在MFGs中学习均衡和社交最优的文献。我们首先确定了M中最常见的设置(静态、稳态和进化的)。

    arXiv:2205.12944v3 Announce Type: replace-cross  Abstract: Non-cooperative and cooperative games with a very large number of players have many applications but remain generally intractable when the number of players increases. Introduced by Lasry and Lions, and Huang, Caines and Malham\'e, Mean Field Games (MFGs) rely on a mean-field approximation to allow the number of players to grow to infinity. Traditional methods for solving these games generally rely on solving partial or stochastic differential equations with a full knowledge of the model. Recently, Reinforcement Learning (RL) has appeared promising to solve complex problems at scale. The combination of RL and MFGs is promising to solve games at a very large scale both in terms of population size and environment complexity. In this survey, we review the quickly growing recent literature on RL methods to learn equilibria and social optima in MFGs. We first identify the most common settings (static, stationary, and evolutive) of M
    
[^2]: 红队游戏：红队语言模型的博弈论框架

    Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models. (arXiv:2310.00322v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00322](http://arxiv.org/abs/2310.00322)

    本文提出了红队游戏（RTG）框架，利用博弈论分析了红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。同时引入了游戏化红队求解器（GRTS）来提供自动化的红队技术。

    

    可部署的大型语言模型（LLM）必须符合有益和无害性的标准，从而实现LLM输出与人类价值的一致性。红队技术是实现这一标准的关键途径。现有的研究仅依赖于手动红队设计和启发式对抗提示进行漏洞检测和优化。这些方法缺乏严格的数学形式化，限制了在可量化度量和收敛保证下对LLM进行多样攻击策略的探索和优化。在本文中，我们提出了红队游戏（RTG），这是一个通用的无需手动标注的博弈论框架。RTG旨在分析红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。在RTG中，我们提出了具有语义空间多样性度量的游戏化红队求解器（GRTS）。GRTS是一种自动化的红队技术，用于解决红队游戏问题。

    Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve 
    
[^3]: 多人不完全信息博弈中的贝叶斯对手建模

    Bayesian Opponent Modeling in Multiplayer Imperfect-Information Games. (arXiv:2212.06027v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2212.06027](http://arxiv.org/abs/2212.06027)

    该论文提出了一种针对多人不完全信息博弈的贝叶斯对手建模方法，在三人 Kuhn poker 中应用这种方法可以明显超过所有的代理商，包括准确的纳什均衡策略。

    

    在许多现实世界的情境中，代理商与多个对立代理商进行战略互动，对手可能采用各种策略。对于这样的情境，设计代理的标准方法是计算或逼近相关的博弈理论解，如纳什均衡，然后遵循规定的策略。然而，这样的策略忽略了对手玩游戏的任何观察，这些观察可能表明可以利用的缺点。我们提出了一种多人不完全信息博弈中的对手建模方法，通过重复交互收集对手玩游戏的观察。我们对三人 Kuhn 扑克展开了对许多真实对手和准确的纳什均衡策略的实验，结果表明我们的算法明显优于所有的代理商，包括准确的纳什均衡策略。

    In many real-world settings agents engage in strategic interactions with multiple opposing agents who can employ a wide variety of strategies. The standard approach for designing agents for such settings is to compute or approximate a relevant game-theoretic solution concept such as Nash equilibrium and then follow the prescribed strategy. However, such a strategy ignores any observations of opponents' play, which may indicate shortcomings that can be exploited. We present an approach for opponent modeling in multiplayer imperfect-information games where we collect observations of opponents' play through repeated interactions. We run experiments against a wide variety of real opponents and exact Nash equilibrium strategies in three-player Kuhn poker and show that our algorithm significantly outperforms all of the agents, including the exact Nash equilibrium strategies.
    

