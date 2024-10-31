# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Edge Caching Based on Deep Reinforcement Learning and Transfer Learning](https://arxiv.org/abs/2402.14576) | 本文提出了一种基于双深度Q学习的缓存方法，通过半马尔可夫决策过程（SMDP）适应现实场景中随机请求到达的特性，综合考虑各种文件特征。 |

# 详细

[^1]: 基于深度强化学习和迁移学习的边缘缓存

    Edge Caching Based on Deep Reinforcement Learning and Transfer Learning

    [https://arxiv.org/abs/2402.14576](https://arxiv.org/abs/2402.14576)

    本文提出了一种基于双深度Q学习的缓存方法，通过半马尔可夫决策过程（SMDP）适应现实场景中随机请求到达的特性，综合考虑各种文件特征。

    

    本文讨论了网络中冗余数据传输日益挑战的问题。流量激增已经使中继链路和骨干网络承压，促使对边缘路由器的缓存解决方案进行探索。现有工作主要依赖于马尔可夫决策过程（MDP）处理缓存问题，假设固定时间间隔的决策；然而，现实场景涉及随机请求到达，尽管各种文件特征在确定最佳缓存策略方面起着至关重要的作用，但相关的现有工作并未考虑所有这些文件特征来形成缓存策略。在本文中，首先我们利用半马尔可夫决策过程（SMDP）来建模缓存问题，以适应现实场景的连续时间特性，允许在文件请求时随机进行缓存决策。然后，我们提出了一种基于双深度Q学习的缓存方法，全面考虑了不同文件特征的影响。

    arXiv:2402.14576v1 Announce Type: cross  Abstract: This paper addresses the escalating challenge of redundant data transmission in networks. The surge in traffic has strained backhaul links and backbone networks, prompting the exploration of caching solutions at the edge router. Existing work primarily relies on Markov Decision Processes (MDP) for caching issues, assuming fixed-time interval decisions; however, real-world scenarios involve random request arrivals, and despite the critical role of various file characteristics in determining an optimal caching policy, none of the related existing work considers all these file characteristics in forming a caching policy. In this paper, first, we formulate the caching problem using a semi-Markov Decision Process (SMDP) to accommodate the continuous-time nature of real-world scenarios allowing for caching decisions at random times upon file requests. Then, we propose a double deep Q-learning-based caching approach that comprehensively accou
    

