# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Update Monte Carlo tree search (UMCTS) algorithm for heuristic global search of sizing optimization problems for truss structures.](http://arxiv.org/abs/2309.06045) | 本文提出了一种基于启发式全局搜索的算法（UMCTS）用于桁架结构尺寸优化问题，通过结合更新过程和蒙特卡洛树搜索（MCTS）以及使用上界置信度（UCB）来获得合适的设计方案。 |

# 详细

[^1]: 基于启发式全局搜索的桁架结构尺寸优化问题的改进蒙特卡洛树搜索（UMCTS）算法

    Update Monte Carlo tree search (UMCTS) algorithm for heuristic global search of sizing optimization problems for truss structures. (arXiv:2309.06045v1 [cs.AI])

    [http://arxiv.org/abs/2309.06045](http://arxiv.org/abs/2309.06045)

    本文提出了一种基于启发式全局搜索的算法（UMCTS）用于桁架结构尺寸优化问题，通过结合更新过程和蒙特卡洛树搜索（MCTS）以及使用上界置信度（UCB）来获得合适的设计方案。

    

    桁架结构尺寸优化是一个复杂的计算问题，强化学习（RL）适用于处理无梯度计算的多模态问题。本研究提出了一种新的高效优化算法——更新蒙特卡洛树搜索（UMCTS），用于获得合适的桁架结构设计。UMCTS是一种基于RL的方法，将新颖的更新过程和蒙特卡洛树搜索（MCTS）与上界置信度（UCB）相结合。更新过程意味着在每一轮中，每个构件的最佳截面积通过搜索树确定，其初始状态是上一轮的最终状态。在UMCTS算法中，引入了加速选择成员面积和迭代次数的加速器，以减少计算时间。此外，对于每个状态，平均奖励被最佳奖励的模拟过程中收集来的奖励替代，确定最优解。本文提出了一种优化算法，通过结合更新过程和MCTS，以及使用UCB来优化桁架结构设计。

    Sizing optimization of truss structures is a complex computational problem, and the reinforcement learning (RL) is suitable for dealing with multimodal problems without gradient computations. In this paper, a new efficient optimization algorithm called update Monte Carlo tree search (UMCTS) is developed to obtain the appropriate design for truss structures. UMCTS is an RL-based method that combines the novel update process and Monte Carlo tree search (MCTS) with the upper confidence bound (UCB). Update process means that in each round, the optimal cross-sectional area of each member is determined by search tree, and its initial state is the final state in the previous round. In the UMCTS algorithm, an accelerator for the number of selections for member area and iteration number is introduced to reduce the computation time. Moreover, for each state, the average reward is replaced by the best reward collected on the simulation process to determine the optimal solution. The proposed optim
    

