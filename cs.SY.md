# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?](https://arxiv.org/abs/2212.14511) | 该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。 |
| [^2] | [Understanding the Application of Utility Theory in Robotics and Artificial Intelligence: A Survey.](http://arxiv.org/abs/2306.09445) | 本文是一项了解机器人和人工智能中效用理论应用的调查，探讨了如何通过合适的效用模型指导智能体选择合理策略来实现系统的最优效用和保证每个群体成员的可持续发展。 |

# 详细

[^1]: 直接潜在模型学习能够解决线性二次高斯控制问题吗？

    Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?

    [https://arxiv.org/abs/2212.14511](https://arxiv.org/abs/2212.14511)

    该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。

    

    我们研究了从潜在高维观测中学习状态表示的任务，目标是控制未知的部分可观察系统。我们采用直接潜在模型学习方法，通过预测与规划直接相关的数量（例如成本）来学习潜在状态空间中的动态模型，而无需重建观测。具体来说，我们专注于一种直观的基于成本驱动的状态表示学习方法，用于解决线性二次高斯（LQG）控制问题，这是最基本的部分可观察控制问题之一。作为我们的主要结果，我们建立了在有限样本下找到近似最优状态表示函数和使用直接学习的潜在模型找到近似最优控制器的保证。据我们所知，尽管以前的相关工作取得了各种经验成功，但在这项工作之前，尚不清楚这种基于成本驱动的潜在模型学习方法是否具有有限样本保证。

    arXiv:2212.14511v2 Announce Type: replace  Abstract: We study the task of learning state representations from potentially high-dimensional observations, with the goal of controlling an unknown partially observable system. We pursue a direct latent model learning approach, where a dynamic model in some latent state space is learned by predicting quantities directly related to planning (e.g., costs) without reconstructing the observations. In particular, we focus on an intuitive cost-driven state representation learning method for solving Linear Quadratic Gaussian (LQG) control, one of the most fundamental partially observable control problems. As our main results, we establish finite-sample guarantees of finding a near-optimal state representation function and a near-optimal controller using the directly learned latent model. To the best of our knowledge, despite various empirical successes, prior to this work it was unclear if such a cost-driven latent model learner enjoys finite-sampl
    
[^2]: 了解效用理论在机器人和人工智能中的应用：一项调查

    Understanding the Application of Utility Theory in Robotics and Artificial Intelligence: A Survey. (arXiv:2306.09445v1 [cs.RO])

    [http://arxiv.org/abs/2306.09445](http://arxiv.org/abs/2306.09445)

    本文是一项了解机器人和人工智能中效用理论应用的调查，探讨了如何通过合适的效用模型指导智能体选择合理策略来实现系统的最优效用和保证每个群体成员的可持续发展。

    

    作为经济学、博弈论和运筹学中的一个统一概念，效用在机器人和人工智能领域中被用来评估个体需求、偏好和利益水平。特别是在多智能体/机器人系统（MAS/MRS）的决策和学习中，合适的效用模型可以指导智能体选择合理的策略来实现其当前需求并学会合作和组织其行为，优化系统的效用，建立稳定可靠的关系，并保证每个群体成员的可持续发展，类似于人类社会。虽然这些系统的复杂、大规模和长期的行为很大程度上由底层关系的基本特性决定，但在机器人和人工智能领域，对机制的理论方面和应用领域的讨论较少。本文引入了一个以效用为导向的需求范式，描述和评估了内部和外部关系。

    As a unifying concept in economics, game theory, and operations research, even in the Robotics and AI field, the utility is used to evaluate the level of individual needs, preferences, and interests. Especially for decision-making and learning in multi-agent/robot systems (MAS/MRS), a suitable utility model can guide agents in choosing reasonable strategies to achieve their current needs and learning to cooperate and organize their behaviors, optimizing the system's utility, building stable and reliable relationships, and guaranteeing each group member's sustainable development, similar to the human society. Although these systems' complex, large-scale, and long-term behaviors are strongly determined by the fundamental characteristics of the underlying relationships, there has been less discussion on the theoretical aspects of mechanisms and the fields of applications in Robotics and AI. This paper introduces a utility-orient needs paradigm to describe and evaluate inter and outer rela
    

