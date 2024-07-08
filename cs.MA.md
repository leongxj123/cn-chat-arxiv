# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ariadne and Theseus: Exploration and Rendezvous with Two Mobile Agents in an Unknown Graph](https://arxiv.org/abs/2403.07748) | 论文研究了移动计算中的探索和会合两个基本问题，提出了分别能在图中$m$个同步时间步实现集体探索和$\frac{3}{2}m$时间步内实现会合的算法。 |
| [^2] | [A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications.](http://arxiv.org/abs/2401.06308) | 本文提出了一种语义感知多址访问方案，旨在优化资源利用与公平性的权衡，并考虑用户数据的相关性，以满足未来6G应用的要求和特性。 |

# 详细

[^1]: 阿瑞阿德涅和忒修斯：在未知图中探索和会合的两个移动代理

    Ariadne and Theseus: Exploration and Rendezvous with Two Mobile Agents in an Unknown Graph

    [https://arxiv.org/abs/2403.07748](https://arxiv.org/abs/2403.07748)

    论文研究了移动计算中的探索和会合两个基本问题，提出了分别能在图中$m$个同步时间步实现集体探索和$\frac{3}{2}m$时间步内实现会合的算法。

    

    我们研究移动计算中的两个基本问题：探索和会合，涉及到一个未知图中的两个不同移动代理。这两个代理可以在所有节点上的白板上读写信息。它们每一步都沿着一个相邻的边移动。在探索问题中，两个代理从图中相同的节点出发，必须遍历所有的边。我们展示了深度优先搜索的一个简单变体可以在$m$个同步时间步中实现集体探索，其中$m$是图的边数。这提高了集体图探索的竞争比率。在会合问题中，代理从图中不同的节点出发，必须尽快相遇。我们介绍了一个算法，保证在至多$\frac{3}{2}m$个时间步内会合。这比所谓的“等妈妈”算法需求的$2m$时间步更好。我们所有的保证都是

    arXiv:2403.07748v1 Announce Type: cross  Abstract: We investigate two fundamental problems in mobile computing: exploration and rendezvous, with two distinct mobile agents in an unknown graph. The agents can read and write information on whiteboards that are located at all nodes. They both move along one adjacent edge at every time-step. In the exploration problem, both agents start from the same node of the graph and must traverse all of its edges. We show that a simple variant of depth-first search achieves collective exploration in $m$ synchronous time-steps, where $m$ is the number of edges of the graph. This improves the competitive ratio of collective graph exploration. In the rendezvous problem, the agents start from different nodes of the graph and must meet as fast as possible. We introduce an algorithm guaranteeing rendezvous in at most $\frac{3}{2}m$ time-steps. This improves over the so-called `wait for Mommy' algorithm which requires $2m$ time-steps. All our guarantees are
    
[^2]: 一种用于分布式、动态6G应用的语义感知多址访问方案

    A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications. (arXiv:2401.06308v1 [cs.NI])

    [http://arxiv.org/abs/2401.06308](http://arxiv.org/abs/2401.06308)

    本文提出了一种语义感知多址访问方案，旨在优化资源利用与公平性的权衡，并考虑用户数据的相关性，以满足未来6G应用的要求和特性。

    

    语义感知范式的出现为创新的服务提供了机会，尤其是在基于6G的应用环境中。尽管在语义提取技术方面取得了显著进展，但将语义信息纳入资源分配决策仍处于早期阶段，缺乏对未来系统需求和特性的考虑。为此，本文引入了一种新的无线频谱多址访问问题的建模。它旨在优化利用率与公平性的权衡，使用α-公平度量，并通过引入自助吞吐量和协助吞吐量的概念来考虑用户数据的相关性。首先，分析了该问题，找出了最优解。接下来，提出了一种基于模型无关的多主体深度强化学习技术的语义感知多智能体双重和决斗深度Q学习 (SAMA-D3QL) 方法。

    The emergence of the semantic-aware paradigm presents opportunities for innovative services, especially in the context of 6G-based applications. Although significant progress has been made in semantic extraction techniques, the incorporation of semantic information into resource allocation decision-making is still in its early stages, lacking consideration of the requirements and characteristics of future systems. In response, this paper introduces a novel formulation for the problem of multiple access to the wireless spectrum. It aims to optimize the utilization-fairness trade-off, using the $\alpha$-fairness metric, while accounting for user data correlation by introducing the concepts of self- and assisted throughputs. Initially, the problem is analyzed to identify its optimal solution. Subsequently, a Semantic-Aware Multi-Agent Double and Dueling Deep Q-Learning (SAMA-D3QL) technique is proposed. This method is grounded in Model-free Multi-Agent Deep Reinforcement Learning (MADRL),
    

