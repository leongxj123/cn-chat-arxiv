# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pointer Networks with Q-Learning for OP Combinatorial Optimization](https://arxiv.org/abs/2311.02629) | 提出了Pointer Q-Network (PQN)方法，将Ptr-Nets和Q-learning相结合，利用其批评者性质，出色地捕获了嵌入图中的关系，从而有效解决了OP组合优化中的具体挑战 |
| [^2] | [ShadowNet for Data-Centric Quantum System Learning.](http://arxiv.org/abs/2308.11290) | 本研究提出了一个数据为中心的量子系统学习范式，将神经网络和经典阴影相结合，以解决大型量子系统动力学的预测和泛化问题。 |
| [^3] | [ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription.](http://arxiv.org/abs/2307.15691) | ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。 |
| [^4] | [Dynamic Pricing and Learning with Bayesian Persuasion.](http://arxiv.org/abs/2304.14385) | 本研究提出了一种计算有效的在线算法，在没有先验知识的情况下，自适应学习最优定价和广告策略，达到次线性后悔。 |
| [^5] | [Doubly robust nearest neighbors in factor models.](http://arxiv.org/abs/2211.14297) | 该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。 |

# 详细

[^1]: 带有Q-Learning的指针网络用于OP组合优化

    Pointer Networks with Q-Learning for OP Combinatorial Optimization

    [https://arxiv.org/abs/2311.02629](https://arxiv.org/abs/2311.02629)

    提出了Pointer Q-Network (PQN)方法，将Ptr-Nets和Q-learning相结合，利用其批评者性质，出色地捕获了嵌入图中的关系，从而有效解决了OP组合优化中的具体挑战

    

    Orienteering Problem（OP）在组合优化（CO）中提出了独特挑战，突显了其在物流、交付和运输规划中的广泛应用。由于OP的NP-hard性质，获得最优解本质上是复杂的。尽管指针网络（Ptr-Nets）在各种组合任务中表现出色，但它们在OP上的表现以及需要专注于未来回报或探索的任务方面仍有改进空间。认识到将强化学习（RL）方法与序列-序列模型相结合的潜能，本研究揭示了指针Q网络（PQN）。该方法结合了Ptr-Nets和Q-learning，由于其仅具批评者性质，它在捕获嵌入图中的关系方面表现出色，这是有效应对OP提出的具体挑战的基本要求。我们探讨了架构和功能

    arXiv:2311.02629v2 Announce Type: replace  Abstract: The Orienteering Problem (OP) presents a unique challenge in Combinatorial Optimization (CO), emphasized by its widespread use in logistics, delivery, and transportation planning. Given the NP-hard nature of OP, obtaining optimal solutions is inherently complex. While Pointer Networks (Ptr-Nets) have exhibited prowess in various combinatorial tasks, their performance in the context of OP, and duties requiring focus on future return or exploration, leaves room for improvement. Recognizing the potency combining Reinforcement Learning (RL) methods with sequence-to-sequence models, this research unveils the Pointer Q-Network (PQN). This method combines Ptr-Nets and Q-learning, which, thanks to its critic only nature, outstands in its capability of capturing relationships within an embedded graph, a fundamental requirement in order to effectively address the specific challenges presented by OP. We explore the architecture and functionalit
    
[^2]: 数据为中心的量子系统学习的影子网络

    ShadowNet for Data-Centric Quantum System Learning. (arXiv:2308.11290v1 [quant-ph])

    [http://arxiv.org/abs/2308.11290](http://arxiv.org/abs/2308.11290)

    本研究提出了一个数据为中心的量子系统学习范式，将神经网络和经典阴影相结合，以解决大型量子系统动力学的预测和泛化问题。

    

    由于维度诅咒，理解大型量子系统的动力学变得困难。统计学习通过神经网络协议和经典阴影在这个领域提供了新的可能性，然而这两种方法都存在局限性：前者受到预测不确定性的困扰，后者缺乏泛化能力。在这里，我们提出了一个数据为中心的学习范式，结合了这两种方法的优势，以促进多样化的量子系统学习任务。特别地，我们的范式利用了经典阴影和其他易于获取的量子系统信息来创建训练数据集，然后通过神经网络来学习探索的量子系统学习问题的潜在映射规律。利用神经网络的泛化能力，这个范式可以在离线训练，并且在推理阶段能够优秀地预测之前未见过的系统，即使只有很少的状态副本。此外，它还继承了

    Understanding the dynamics of large quantum systems is hindered by the curse of dimensionality. Statistical learning offers new possibilities in this regime by neural-network protocols and classical shadows, while both methods have limitations: the former is plagued by the predictive uncertainty and the latter lacks the generalization ability. Here we propose a data-centric learning paradigm combining the strength of these two approaches to facilitate diverse quantum system learning (QSL) tasks. Particularly, our paradigm utilizes classical shadows along with other easily obtainable information of quantum systems to create the training dataset, which is then learnt by neural networks to unveil the underlying mapping rule of the explored QSL problem. Capitalizing on the generalization power of neural networks, this paradigm can be trained offline and excel at predicting previously unseen systems at the inference stage, even with few state copies. Besides, it inherits the characteristic 
    
[^3]: ODTlearn: 一个用于学习预测和处方的最优决策树的包

    ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription. (arXiv:2307.15691v1 [stat.ML])

    [http://arxiv.org/abs/2307.15691](http://arxiv.org/abs/2307.15691)

    ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。

    

    ODTLearn是一个开源的Python包，提供了基于混合整数优化(MIO)框架的高风险预测和处方任务的最优决策树学习方法。该包的当前版本提供了学习最优分类树、公平最优分类树、鲁棒最优分类树和从观测数据学习最优处方树的实现。我们设计了该包以便于维护和扩展，当引入新的最优决策树问题类、重构策略和解决算法时，可以轻松更新。为此，该包遵循面向对象的设计原则，并支持商业(Gurobi)和开源(COIN-OR branch and cut)求解器。包的文档和详细用户指南可以在https://d3m-research-group.github.io/odtlearn/找到。

    ODTLearn is an open-source Python package that provides methods for learning optimal decision trees for high-stakes predictive and prescriptive tasks based on the mixed-integer optimization (MIO) framework proposed in Aghaei et al. (2019) and several of its extensions. The current version of the package provides implementations for learning optimal classification trees, optimal fair classification trees, optimal classification trees robust to distribution shifts, and optimal prescriptive trees from observational data. We have designed the package to be easy to maintain and extend as new optimal decision tree problem classes, reformulation strategies, and solution algorithms are introduced. To this end, the package follows object-oriented design principles and supports both commercial (Gurobi) and open source (COIN-OR branch and cut) solvers. The package documentation and an extensive user guide can be found at https://d3m-research-group.github.io/odtlearn/. Additionally, users can view
    
[^4]: 带有贝叶斯说服的动态定价和学习

    Dynamic Pricing and Learning with Bayesian Persuasion. (arXiv:2304.14385v1 [cs.GT])

    [http://arxiv.org/abs/2304.14385](http://arxiv.org/abs/2304.14385)

    本研究提出了一种计算有效的在线算法，在没有先验知识的情况下，自适应学习最优定价和广告策略，达到次线性后悔。

    

    我们考虑了一个新颖的动态定价和学习设置，在按顺序设置产品价格的同时，卖家还预先承诺“广告方案”。也就是说，在每轮开始时，卖家可以决定提供什么样的信号来告知买家产品实际的质量。我们使用流行的贝叶斯说服框架来模拟这些信号对买家的评估和购买反应的影响，我们制定了在最大化卖方预期收入的同时找到广告方案和定价方案的最优设计问题。在没有任何先验知识的情况下，我们的目标是设计一个在线算法，该算法可以使用过去的购买反应来自适应地学习最优定价和广告策略。我们研究了算法的后悔，与最优的千里之堤价格和广告计划进行比较。我们的主要结果是一种计算有效的在线算法，即使卖家没有买家需求函数的先验知识，也可以实现与最佳固定价格和广告方案相关的次线性后悔。

    We consider a novel dynamic pricing and learning setting where in addition to setting prices of products in sequential rounds, the seller also ex-ante commits to 'advertising schemes'. That is, in the beginning of each round the seller can decide what kind of signal they will provide to the buyer about the product's quality upon realization. Using the popular Bayesian persuasion framework to model the effect of these signals on the buyers' valuation and purchase responses, we formulate the problem of finding an optimal design of the advertising scheme along with a pricing scheme that maximizes the seller's expected revenue. Without any apriori knowledge of the buyers' demand function, our goal is to design an online algorithm that can use past purchase responses to adaptively learn the optimal pricing and advertising strategy. We study the regret of the algorithm when compared to the optimal clairvoyant price and advertising scheme.  Our main result is a computationally efficient onlin
    
[^5]: 因子模型中的双重稳健最近邻方法

    Doubly robust nearest neighbors in factor models. (arXiv:2211.14297v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14297](http://arxiv.org/abs/2211.14297)

    该论文介绍了一种在潜在因子模型中处理缺失数据的双重稳健最近邻方法，可以提供一致的估计，并在存在良好的行和列邻居时提供（近似）二次改进非渐近性能。

    

    我们介绍并分析了在潜在因子模型中处理缺失数据的改进最近邻（NN）方法。我们考虑一个带有缺失数据的矩阵补全问题，其中当被观察到时，第$(i, t)$个条目由其均值$f(u_i, v_t)$加上均值为零的噪声给出，其中$f$为未知函数，$u_i$和$v_t$为潜在因子。之前的NN策略，如单元-单元NN，用于估计均值$f(u_i, v_t)$，依赖于存在其他行$j$使得$u_j \approx u_i$。类似地，时间-时间NN策略依赖于存在列$t'$使得$v_{t'} \approx v_t$。当相似行或相似列不可用时，这些策略的性能较差。我们的估计在两个方面对这种不足是双重稳健的：(1) 只要存在良好的行或列邻居，我们的估计提供一致的估计。 (2) 此外，如果存在良好的行和列邻居，它提供了（近似）二次改进非渐近性能。

    We introduce and analyze an improved variant of nearest neighbors (NN) for estimation with missing data in latent factor models. We consider a matrix completion problem with missing data, where the $(i, t)$-th entry, when observed, is given by its mean $f(u_i, v_t)$ plus mean-zero noise for an unknown function $f$ and latent factors $u_i$ and $v_t$. Prior NN strategies, like unit-unit NN, for estimating the mean $f(u_i, v_t)$ relies on existence of other rows $j$ with $u_j \approx u_i$. Similarly, time-time NN strategy relies on existence of columns $t'$ with $v_{t'} \approx v_t$. These strategies provide poor performance respectively when similar rows or similar columns are not available. Our estimate is doubly robust to this deficit in two ways: (1) As long as there exist either good row or good column neighbors, our estimate provides a consistent estimate. (2) Furthermore, if both good row and good column neighbors exist, it provides a (near-)quadratic improvement in the non-asympto
    

