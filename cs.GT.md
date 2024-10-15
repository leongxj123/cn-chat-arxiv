# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regret Minimization in Stackelberg Games with Side Information](https://arxiv.org/abs/2402.08576) | 这篇论文研究了侧信息中的Stackelberg博弈，提出了一种方法来解决现实中玩家之间信息交流不充分的情况，并且证明了在这种情况下后悔最小化是有效的。 |
| [^2] | [Active Inverse Learning in Stackelberg Trajectory Games.](http://arxiv.org/abs/2308.08017) | 这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。 |
| [^3] | [Optimal Scoring Rule Design under Partial Knowledge.](http://arxiv.org/abs/2107.07420) | 本文研究了在委托人对代理人的信号分布部分了解的情况下，最优打分规则的设计问题。作者提出了一个最大最小优化的框架，来最大化在代理人信号分布的集合中最坏情况下回报的增加。对于有限集合，提出了高效的算法；对于无限集合，提出了完全多项式时间逼近方案。 |

# 详细

[^1]: 侧信息中的Stackelberg博弈中的后悔最小化

    Regret Minimization in Stackelberg Games with Side Information

    [https://arxiv.org/abs/2402.08576](https://arxiv.org/abs/2402.08576)

    这篇论文研究了侧信息中的Stackelberg博弈，提出了一种方法来解决现实中玩家之间信息交流不充分的情况，并且证明了在这种情况下后悔最小化是有效的。

    

    在最基本的情况下，Stackelberg博弈是一个双人博弈，其中领导者承诺一种（混合）策略，追随者做出最佳反应。在过去的十年中，Stackelberg博弈算法是算法博弈论的最大成功之一，因为Stackelberg博弈的算法已经在许多现实世界的领域中被应用，包括机场安全、反盗猎和网络犯罪预防。然而，这些算法通常未能考虑到每个玩家可用的额外信息（例如交通模式，天气条件，网络拥塞），这是现实的显著特征，可能会显著影响到两个玩家的最优策略。我们将这样的情况形式化为带有侧信息的Stackelberg博弈，其中两个玩家在进行游戏之前都观察到一个外部环境。然后，领导者承诺一种（可能依赖于上下文的）策略，追随者对领导者的策略和上下文都做出最佳反应。

    In its most basic form, a Stackelberg game is a two-player game in which a leader commits to a (mixed) strategy, and a follower best-responds. Stackelberg games are perhaps one of the biggest success stories of algorithmic game theory over the last decade, as algorithms for playing in Stackelberg games have been deployed in many real-world domains including airport security, anti-poaching efforts, and cyber-crime prevention. However, these algorithms often fail to take into consideration the additional information available to each player (e.g. traffic patterns, weather conditions, network congestion), a salient feature of reality which may significantly affect both players' optimal strategies. We formalize such settings as Stackelberg games with side information, in which both players observe an external context before playing. The leader then commits to a (possibly context-dependent) strategy, and the follower best-responds to both the leader's strategy and the context. We focus on t
    
[^2]: Stackelberg轨迹博弈中的主动逆向学习

    Active Inverse Learning in Stackelberg Trajectory Games. (arXiv:2308.08017v1 [cs.GT])

    [http://arxiv.org/abs/2308.08017](http://arxiv.org/abs/2308.08017)

    这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。

    

    博弈论的逆向学习是从玩家的行为中推断出他们的目标的问题。我们在一个Stackelberg博弈中，通过每个玩家的动态系统轨迹来定义一个逆向学习问题，其中包括一个领导者和一个跟随者。我们提出了一种主动逆向学习方法，让领导者推断出一个有限候选集中描述跟随者目标函数的假设。与现有方法使用被动观察到的轨迹不同，所提出的方法主动地最大化不同假设下跟随者轨迹的差异，加速领导者的推断过程。我们在一个递进的重复轨迹博弈中展示了所提出的方法。与均匀随机输入相比，所提供的方法加速了概率收敛到条件于跟随者轨迹的不同假设上的收敛速度。

    Game-theoretic inverse learning is the problem of inferring the players' objectives from their actions. We formulate an inverse learning problem in a Stackelberg game between a leader and a follower, where each player's action is the trajectory of a dynamical system. We propose an active inverse learning method for the leader to infer which hypothesis among a finite set of candidates describes the follower's objective function. Instead of using passively observed trajectories like existing methods, the proposed method actively maximizes the differences in the follower's trajectories under different hypotheses to accelerate the leader's inference. We demonstrate the proposed method in a receding-horizon repeated trajectory game. Compared with uniformly random inputs, the leader inputs provided by the proposed method accelerate the convergence of the probability of different hypotheses conditioned on the follower's trajectory by orders of magnitude.
    
[^3]: 部分知识下的最优打分规则设计

    Optimal Scoring Rule Design under Partial Knowledge. (arXiv:2107.07420v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2107.07420](http://arxiv.org/abs/2107.07420)

    本文研究了在委托人对代理人的信号分布部分了解的情况下，最优打分规则的设计问题。作者提出了一个最大最小优化的框架，来最大化在代理人信号分布的集合中最坏情况下回报的增加。对于有限集合，提出了高效的算法；对于无限集合，提出了完全多项式时间逼近方案。

    

    本文研究了当委托人对代理人的信号分布部分了解时，最优适当打分规则的设计。最近的工作表明，在委托人完全了解代理人的信号分布的假设下，可以确定增加代理人回报的最大适当打分规则，当代理人选择访问昂贵信号以完善其先验预测的后验信念时。在我们的设置中，委托人只知道代理人的信号分布属于一组分布中的某个。我们将打分规则设计问题制定为最大最小优化问题，最大化分布集合中最坏情况下回报的增加。当分布集合有限时，我们提出了一种高效的算法来计算最优打分规则，并设计了一种完全多项式时间逼近方案，适用于各种无限集合的分布。我们进一步指出，广泛使用的打分规则，如二次方打分规则。

    This paper studies the design of optimal proper scoring rules when the principal has partial knowledge of an agent's signal distribution. Recent work characterizes the proper scoring rules that maximize the increase of an agent's payoff when the agent chooses to access a costly signal to refine a posterior belief from her prior prediction, under the assumption that the agent's signal distribution is fully known to the principal. In our setting, the principal only knows about a set of distributions where the agent's signal distribution belongs. We formulate the scoring rule design problem as a max-min optimization that maximizes the worst-case increase in payoff across the set of distributions.  We propose an efficient algorithm to compute an optimal scoring rule when the set of distributions is finite, and devise a fully polynomial-time approximation scheme that accommodates various infinite sets of distributions. We further remark that widely used scoring rules, such as the quadratic 
    

