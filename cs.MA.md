# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space](https://arxiv.org/abs/2404.01752) | 提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案 |
| [^2] | [Locally Differentially Private Distributed Online Learning with Guaranteed Optimality.](http://arxiv.org/abs/2306.14094) | 本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。 |

# 详细

[^1]: 安全间隔RRT*用于连续空间中可扩展的多机器人路径规划

    Safe Interval RRT* for Scalable Multi-Robot Path Planning in Continuous Space

    [https://arxiv.org/abs/2404.01752](https://arxiv.org/abs/2404.01752)

    提出了安全间隔RRT*（SI-RRT*）两级方法，低级采用采样规划器找到单个机器人的无碰撞轨迹，高级通过优先规划或基于冲突的搜索解决机器人间冲突，实验结果表明SI-RRT*能够快速找到高质量解决方案

    

    在本文中，我们考虑了在连续空间中解决多机器人路径规划（MRPP）问题以找到无冲突路径的问题。问题的困难主要来自两个因素。首先，涉及多个机器人会导致组合决策，使搜索空间呈指数级增长。其次，连续空间呈现出潜在无限的状态和动作。针对这个问题，我们提出了一个两级方法，低级是基于采样的规划器安全间隔RRT*（SI-RRT*），用于找到单个机器人的无碰撞轨迹。高级可以使用能够解决机器人间冲突的任何方法，我们采用了两种代表性方法，即优先规划（SI-CPP）和基于冲突的搜索（SI-CCBS）。实验结果表明，SI-RRT*能够快速找到高质量解决方案，并且所需的样本数量较少。SI-CPP表现出更好的可扩展性，而SI-CCBS产生

    arXiv:2404.01752v1 Announce Type: cross  Abstract: In this paper, we consider the problem of Multi-Robot Path Planning (MRPP) in continuous space to find conflict-free paths. The difficulty of the problem arises from two primary factors. First, the involvement of multiple robots leads to combinatorial decision-making, which escalates the search space exponentially. Second, the continuous space presents potentially infinite states and actions. For this problem, we propose a two-level approach where the low level is a sampling-based planner Safe Interval RRT* (SI-RRT*) that finds a collision-free trajectory for individual robots. The high level can use any method that can resolve inter-robot conflicts where we employ two representative methods that are Prioritized Planning (SI-CPP) and Conflict Based Search (SI-CCBS). Experimental results show that SI-RRT* can find a high-quality solution quickly with a small number of samples. SI-CPP exhibits improved scalability while SI-CCBS produces 
    
[^2]: 具有保证最优性的局部差分隐私分布式在线学习

    Locally Differentially Private Distributed Online Learning with Guaranteed Optimality. (arXiv:2306.14094v1 [cs.LG])

    [http://arxiv.org/abs/2306.14094](http://arxiv.org/abs/2306.14094)

    本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。

    

    分布式在线学习由于其处理大规模数据集和流数据的能力而受到越来越多的关注。为了解决隐私保护问题，已经提出了许多个人私密分布式在线学习算法，大多数基于差分隐私，差分隐私已成为隐私保护的“黄金标准”。然而，这些算法常常面临为了隐私保护而牺牲学习准确性的困境。本文利用在线学习的独特特征，提出了一种方法来解决这一困境，并确保分布式在线学习中的差分隐私和学习准确性。具体而言，该方法在确保预期瞬时遗憾程度逐渐减小的同时，还能保证有限的累积隐私预算，即使在无限时间范围内。为了应对完全分布式环境，我们采用本地差分隐私框架，避免了对全局数据的依赖。

    Distributed online learning is gaining increased traction due to its unique ability to process large-scale datasets and streaming data. To address the growing public awareness and concern on privacy protection, plenty of private distributed online learning algorithms have been proposed, mostly based on differential privacy which has emerged as the ``gold standard" for privacy protection. However, these algorithms often face the dilemma of trading learning accuracy for privacy. By exploiting the unique characteristics of online learning, this paper proposes an approach that tackles the dilemma and ensures both differential privacy and learning accuracy in distributed online learning. More specifically, while ensuring a diminishing expected instantaneous regret, the approach can simultaneously ensure a finite cumulative privacy budget, even on the infinite time horizon. To cater for the fully distributed setting, we adopt the local differential-privacy framework which avoids the reliance
    

