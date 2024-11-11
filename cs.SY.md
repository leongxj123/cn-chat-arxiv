# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Entropy-regularized Diffusion Policy with Q-Ensembles for Offline Reinforcement Learning](https://arxiv.org/abs/2402.04080) | 本文介绍了一种熵正则化的扩散策略与Q-集合相结合的离线强化学习方法，该方法通过将一个复杂的动作分布转化为标准高斯分布，然后使用逆时间SDE采样动作，以改善离线数据集的探索能力，并通过学习Q-集合的下信心界实现更强健的策略改进。在D4RL基准任务的大多数任务上达到了最先进的性能。 |

# 详细

[^1]: 熵正则化扩散策略与Q-集合用于离线强化学习

    Entropy-regularized Diffusion Policy with Q-Ensembles for Offline Reinforcement Learning

    [https://arxiv.org/abs/2402.04080](https://arxiv.org/abs/2402.04080)

    本文介绍了一种熵正则化的扩散策略与Q-集合相结合的离线强化学习方法，该方法通过将一个复杂的动作分布转化为标准高斯分布，然后使用逆时间SDE采样动作，以改善离线数据集的探索能力，并通过学习Q-集合的下信心界实现更强健的策略改进。在D4RL基准任务的大多数任务上达到了最先进的性能。

    

    本文提出了训练用于离线强化学习的扩散策略的先进技术。核心是一个均值回归随机微分方程（SDE），它将复杂的动作分布转化为标准高斯分布，然后在环境状态条件下使用相应的逆时间SDE采样动作，类似于典型的扩散策略。我们展示了这样一个SDE有解，我们可以用它来计算策略的对数概率，从而得到一个熵正则项，改进了离线数据集的探索能力。为了减轻来自分布外数据点的不准确值函数的影响，我们进一步提出了学习Q-集合的下信心界以实现更强健的策略改进。通过将熵正则化扩散策略与Q-集合结合应用于离线强化学习，我们的方法在D4RL基准任务的大多数任务上达到了最先进的性能。代码可在\href{https://github.com/ruoqizzz/Entro}{https://github.com/ruoqizzz/Entro}找到。

    This paper presents advanced techniques of training diffusion policies for offline reinforcement learning (RL). At the core is a mean-reverting stochastic differential equation (SDE) that transfers a complex action distribution into a standard Gaussian and then samples actions conditioned on the environment state with a corresponding reverse-time SDE, like a typical diffusion policy. We show that such an SDE has a solution that we can use to calculate the log probability of the policy, yielding an entropy regularizer that improves the exploration of offline datasets. To mitigate the impact of inaccurate value functions from out-of-distribution data points, we further propose to learn the lower confidence bound of Q-ensembles for more robust policy improvement. By combining the entropy-regularized diffusion policy with Q-ensembles in offline RL, our method achieves state-of-the-art performance on most tasks in D4RL benchmarks. Code is available at \href{https://github.com/ruoqizzz/Entro
    

