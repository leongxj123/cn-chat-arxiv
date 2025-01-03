# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Solving Hierarchical Information-Sharing Dec-POMDPs: An Extensive-Form Game Approach](https://arxiv.org/abs/2402.02954) | 本文通过应用最优性原理研究了分层信息共享的分布式部分可观察马尔可夫决策过程的解决方法。通过将问题分解成单阶段子游戏，并通过进一步分解子游戏，我们成功地解开了决策变量的纠缠，同时显著减少了时间复杂度。 |
| [^2] | [Fine-Tuning Games: Bargaining and Adaptation for General-Purpose Models.](http://arxiv.org/abs/2308.04399) | 本文提出了一个模型，探讨了通用模型的微调过程中的利润分享问题，为一般类的成本和收入函数描述了解决方案的条件。 |
| [^3] | [The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents.](http://arxiv.org/abs/2305.16695) | 本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。 |
| [^4] | [Robust Auction Design with Support Information.](http://arxiv.org/abs/2305.09065) | 本文提出了一个新颖的带支持信息的拍卖设计，通过优化DSIC机制并将最坏情况与oracle进行比较，讲述了三种支持信息的区域，得出了最优机制的闭合形式。 |

# 详细

[^1]: 解决分层信息共享的分布式部分可观察马尔可夫决策过程：一种广义博弈方法

    Solving Hierarchical Information-Sharing Dec-POMDPs: An Extensive-Form Game Approach

    [https://arxiv.org/abs/2402.02954](https://arxiv.org/abs/2402.02954)

    本文通过应用最优性原理研究了分层信息共享的分布式部分可观察马尔可夫决策过程的解决方法。通过将问题分解成单阶段子游戏，并通过进一步分解子游戏，我们成功地解开了决策变量的纠缠，同时显著减少了时间复杂度。

    

    最近的理论表明，多人分散的部分可观察马尔可夫决策过程可以转化为等效的单人游戏，使得可以应用贝尔曼的最优性原理通过将其分解为单阶段子游戏来解决单人游戏。然而，这种方法在每个单阶段子游戏中纠缠了所有玩家的决策变量，导致指数复杂度的备份。本文展示了如何在保持分层信息共享的前提下解开这些决策变量的纠缠，这是我们社会中一种突出的管理风格。为了实现这个目标，我们应用最优性原理通过进一步将任何单阶段子游戏分解为更小的子游戏来解决它，使我们能够逐次进行单人决策。我们的方法揭示了存在于单阶段子游戏中的广义博弈解决方案，极大地减少了时间复杂度。我们的实验结果验证了我们方法的有效性，证明它可以在解决分层信息共享的分布式部分可观察马尔可夫决策过程中发挥重要作用。

    A recent theory shows that a multi-player decentralized partially observable Markov decision process can be transformed into an equivalent single-player game, enabling the application of \citeauthor{bellman}'s principle of optimality to solve the single-player game by breaking it down into single-stage subgames. However, this approach entangles the decision variables of all players at each single-stage subgame, resulting in backups with a double-exponential complexity. This paper demonstrates how to disentangle these decision variables while maintaining optimality under hierarchical information sharing, a prominent management style in our society. To achieve this, we apply the principle of optimality to solve any single-stage subgame by breaking it down further into smaller subgames, enabling us to make single-player decisions at a time. Our approach reveals that extensive-form games always exist with solutions to a single-stage subgame, significantly reducing time complexity. Our expe
    
[^2]: Fine-Tuning Games: Bargaining and Adaptation for General-Purpose Models

    Fine-Tuning Games: Bargaining and Adaptation for General-Purpose Models. (arXiv:2308.04399v1 [cs.GT])

    [http://arxiv.org/abs/2308.04399](http://arxiv.org/abs/2308.04399)

    本文提出了一个模型，探讨了通用模型的微调过程中的利润分享问题，为一般类的成本和收入函数描述了解决方案的条件。

    

    机器学习（ML）和人工智能（AI）方面的重大进展越来越多地采用开发和发布通用模型的形式。这些模型旨在由其他企业和机构进行适应，以执行特定的领域专用功能。这个过程被称为适应或微调。本文提供了一个微调过程的模型，其中一位通用专家将技术产品（即ML模型）提升到一定的性能水平，并且一位或多位领域专家将其调整适用于特定领域。这两个实体都是追求利润的，当他们投资于技术时会产生成本，在技术进入市场前，他们必须就如何分享收入达成谈判协议。对于相对一般的成本和收入函数类，我们刻画了微调博弈产生利润分享解决方案的条件。我们观察到，任何潜在的领域专业化都会产生...

    Major advances in Machine Learning (ML) and Artificial Intelligence (AI) increasingly take the form of developing and releasing general-purpose models. These models are designed to be adapted by other businesses and agencies to perform a particular, domain-specific function. This process has become known as adaptation or fine-tuning. This paper offers a model of the fine-tuning process where a Generalist brings the technological product (here an ML model) to a certain level of performance, and one or more Domain-specialist(s) adapts it for use in a particular domain. Both entities are profit-seeking and incur costs when they invest in the technology, and they must reach a bargaining agreement on how to share the revenue for the technology to reach the market. For a relatively general class of cost and revenue functions, we characterize the conditions under which the fine-tuning game yields a profit-sharing solution. We observe that any potential domain-specialization will either contri
    
[^3]: 寻求稳定性：具有初始文件的战略出版商的学习动态的研究

    The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents. (arXiv:2305.16695v1 [cs.GT])

    [http://arxiv.org/abs/2305.16695](http://arxiv.org/abs/2305.16695)

    本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。

    

    我们研究了一种信息检索的博弈论模型，其中战略出版商旨在在保持原始文档完整性的同时最大化自己排名第一的机会。我们表明，常用的PRP排名方案导致环境不稳定，游戏经常无法达到纯纳什均衡。我们将相对排名原则（RRP）作为替代排名原则，并介绍两个排名函数，它们是RRP的实例。我们提供了理论和实证证据，表明这些方法导致稳定的搜索生态系统，通过提供关于学习动力学收敛的积极结果。我们还定义出版商和用户的福利，并展示了可能的出版商-用户权衡，突显了确定搜索引擎设计师应选择哪种排名函数的复杂性。

    We study a game-theoretic model of information retrieval, in which strategic publishers aim to maximize their chances of being ranked first by the search engine, while maintaining the integrity of their original documents. We show that the commonly used PRP ranking scheme results in an unstable environment where games often fail to reach pure Nash equilibrium. We propose the Relative Ranking Principle (RRP) as an alternative ranking principle, and introduce two ranking functions that are instances of the RRP. We provide both theoretical and empirical evidence that these methods lead to a stable search ecosystem, by providing positive results on the learning dynamics convergence. We also define the publishers' and users' welfare, and demonstrate a possible publisher-user trade-off, which highlights the complexity of determining which ranking function should be selected by the search engine designer.
    
[^4]: 带支持信息的鲁棒拍卖设计

    Robust Auction Design with Support Information. (arXiv:2305.09065v1 [econ.TH])

    [http://arxiv.org/abs/2305.09065](http://arxiv.org/abs/2305.09065)

    本文提出了一个新颖的带支持信息的拍卖设计，通过优化DSIC机制并将最坏情况与oracle进行比较，讲述了三种支持信息的区域，得出了最优机制的闭合形式。

    

    一个卖家想要将商品卖给$n$个买家，买家的估值是独立同分布的，但是卖家并不知道这个分布。为了抵御环境和买家行为的不确定性，卖家在DSIC机制中进行优化，并将最坏情况的表现与具有完全买家估值知识的oracle进行比较。我们的分析包括遗憾和比率两个目标。对于这些目标，我们以支持和买家数$n$的函数形式导出了一个闭合的最优机制。我们的分析揭示了三个支持信息的区域和一个新的鲁棒机制类。

    A seller wants to sell an item to $n$ buyers. The buyer valuations are drawn i.i.d. from a distribution, but the seller does not know this distribution; the seller only knows the support $[a,b]$. To be robust against the lack of knowledge of the environment and buyers' behavior, the seller optimizes over DSIC mechanisms, and measures the worst-case performance relative to an oracle with complete knowledge of buyers' valuations. Our analysis encompasses both the regret and the ratio objectives.  For these objectives, we derive an optimal mechanism in closed form as a function of the support and the number of buyers $n$. Our analysis reveals three regimes of support information and a new class of robust mechanisms. i.) With "low" support information, the optimal mechanism is a second-price auction (SPA) with a random reserve, a focal class in the earlier literature. ii.) With "high" support information, we show that second-price auctions are strictly suboptimal, and an optimal mechanism 
    

