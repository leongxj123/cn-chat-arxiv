# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?](https://arxiv.org/abs/2212.14511) | 该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。 |

# 详细

[^1]: 直接潜在模型学习能够解决线性二次高斯控制问题吗？

    Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?

    [https://arxiv.org/abs/2212.14511](https://arxiv.org/abs/2212.14511)

    该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。

    

    我们研究了从潜在高维观测中学习状态表示的任务，目标是控制未知的部分可观察系统。我们采用直接潜在模型学习方法，通过预测与规划直接相关的数量（例如成本）来学习潜在状态空间中的动态模型，而无需重建观测。具体来说，我们专注于一种直观的基于成本驱动的状态表示学习方法，用于解决线性二次高斯（LQG）控制问题，这是最基本的部分可观察控制问题之一。作为我们的主要结果，我们建立了在有限样本下找到近似最优状态表示函数和使用直接学习的潜在模型找到近似最优控制器的保证。据我们所知，尽管以前的相关工作取得了各种经验成功，但在这项工作之前，尚不清楚这种基于成本驱动的潜在模型学习方法是否具有有限样本保证。

    arXiv:2212.14511v2 Announce Type: replace  Abstract: We study the task of learning state representations from potentially high-dimensional observations, with the goal of controlling an unknown partially observable system. We pursue a direct latent model learning approach, where a dynamic model in some latent state space is learned by predicting quantities directly related to planning (e.g., costs) without reconstructing the observations. In particular, we focus on an intuitive cost-driven state representation learning method for solving Linear Quadratic Gaussian (LQG) control, one of the most fundamental partially observable control problems. As our main results, we establish finite-sample guarantees of finding a near-optimal state representation function and a near-optimal controller using the directly learned latent model. To the best of our knowledge, despite various empirical successes, prior to this work it was unclear if such a cost-driven latent model learner enjoys finite-sampl
    

