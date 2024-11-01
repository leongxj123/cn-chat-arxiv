# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |
| [^2] | [Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning.](http://arxiv.org/abs/2309.04459) | 通过将行动空间离散化并采用分词技术，我们提出了一种在稀疏奖励强化学习中生成技巧的新方法。这种方法能够减少探索的难度，并在连续行动空间中达到良好的性能。 |

# 详细

[^1]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    
[^2]: 子词作为技巧：稀疏奖励强化学习的分词化

    Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning. (arXiv:2309.04459v1 [cs.LG])

    [http://arxiv.org/abs/2309.04459](http://arxiv.org/abs/2309.04459)

    通过将行动空间离散化并采用分词技术，我们提出了一种在稀疏奖励强化学习中生成技巧的新方法。这种方法能够减少探索的难度，并在连续行动空间中达到良好的性能。

    

    稀疏奖励强化学习中的探索具有困难，因为需要通过长期的、协调的行动序列才能获得任何奖励。而且，在连续的行动空间中，可能的行动数量是无穷多的，这只会增加探索的难度。为了解决这些问题，一类方法通过在同一领域收集的交互数据中形成时间上延伸的行动，通常称为技巧，并在这个新的行动空间上进行策略的优化。通常这样的方法在连续行动空间中需要一个漫长的预训练阶段，在强化学习开始之前形成技巧。鉴于先前的证据表明在这些任务中并不需要完整的连续行动空间，我们提出了一种新颖的技巧生成方法，包括两个组成部分。首先，我们通过聚类将行动空间离散化，然后我们利用从自然语言处理借鉴来的分词技术。

    Exploration in sparse-reward reinforcement learning is difficult due to the requirement of long, coordinated sequences of actions in order to achieve any reward. Moreover, in continuous action spaces there are an infinite number of possible actions, which only increases the difficulty of exploration. One class of methods designed to address these issues forms temporally extended actions, often called skills, from interaction data collected in the same domain, and optimizes a policy on top of this new action space. Typically such methods require a lengthy pretraining phase, especially in continuous action spaces, in order to form the skills before reinforcement learning can begin. Given prior evidence that the full range of the continuous action space is not required in such tasks, we propose a novel approach to skill-generation with two components. First we discretize the action space through clustering, and second we leverage a tokenization technique borrowed from natural language pro
    

