# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space](https://arxiv.org/abs/2404.01752) | 提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案 |
| [^2] | [(Ir)rationality in AI: State of the Art, Research Challenges and Open Questions](https://arxiv.org/abs/2311.17165) | 这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。 |

# 详细

[^1]: 安全间隔RRT*用于连续空间中可扩展的多机器人路径规划

    Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space

    [https://arxiv.org/abs/2404.01752](https://arxiv.org/abs/2404.01752)

    提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案

    

    在本文中，我们考虑了在连续空间中解决多机器人路径规划（MRPP）问题以找到无冲突路径的问题。问题的困难主要来自两个因素。首先，涉及多个机器人会导致组合决策，使搜索空间呈指数级增长。其次，连续空间呈现出潜在无限的状态和动作。针对这个问题，我们提出了一个两级方法，低级是基于采样的规划器安全间隔RRT*（SI-RRT*），用于找到单个机器人的无碰撞轨迹。高级可以使用能够解决机器人间冲突的任何方法，我们采用了两种代表性方法，即优先规划（SI-CPP）和基于冲突的搜索（SI-CCBS）。实验结果表明，SI-RRT*能够快速找到高质量解决方案，并且所需的样本数量较少。SI-CPP表现出更好的可扩展性，而SI-CCBS产生

    arXiv:2404.01752v1 Announce Type: cross  Abstract: In this paper, we consider the problem of Multi-Robot Path Planning (MRPP) in continuous space to find conflict-free paths. The difficulty of the problem arises from two primary factors. First, the involvement of multiple robots leads to combinatorial decision-making, which escalates the search space exponentially. Second, the continuous space presents potentially infinite states and actions. For this problem, we propose a two-level approach where the low level is a sampling-based planner Safe Interval RRT* (SI-RRT*) that finds a collision-free trajectory for individual robots. The high level can use any method that can resolve inter-robot conflicts where we employ two representative methods that are Prioritized Planning (SI-CPP) and Conflict Based Search (SI-CCBS). Experimental results show that SI-RRT* can find a high-quality solution quickly with a small number of samples. SI-CPP exhibits improved scalability while SI-CCBS produces 
    
[^2]: (非)理性在人工智能中的应用：现状、研究挑战和未解之问

    (Ir)rationality in AI: State of the Art, Research Challenges and Open Questions

    [https://arxiv.org/abs/2311.17165](https://arxiv.org/abs/2311.17165)

    这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。

    

    理性概念在人工智能领域中占据着重要地位。无论是模拟人类推理还是追求有限最优性，我们通常希望使人工智能代理尽可能理性。尽管这个概念在人工智能中非常核心，但对于什么构成理性代理并没有统一的定义。本文调查了人工智能中的理性与非理性，并提出了这个领域的未解问题。在其他领域对理性的理解对其在人工智能中的概念产生了影响，特别是经济学、哲学和心理学方面的研究。着重考虑人工智能代理的行为，我们探讨了在某些情境中非理性行为可能是最优的情况。关于处理非理性代理的方法已经得到了一些发展，包括识别和交互等方面的研究，然而，在这个领域的工作仍然存在一些挑战和问题。

    arXiv:2311.17165v2 Announce Type: replace Abstract: The concept of rationality is central to the field of artificial intelligence. Whether we are seeking to simulate human reasoning, or the goal is to achieve bounded optimality, we generally seek to make artificial agents as rational as possible. Despite the centrality of the concept within AI, there is no unified definition of what constitutes a rational agent. This article provides a survey of rationality and irrationality in artificial intelligence, and sets out the open questions in this area. The understanding of rationality in other fields has influenced its conception within artificial intelligence, in particular work in economics, philosophy and psychology. Focusing on the behaviour of artificial agents, we consider irrational behaviours that can prove to be optimal in certain scenarios. Some methods have been developed to deal with irrational agents, both in terms of identification and interaction, however work in this area re
    

