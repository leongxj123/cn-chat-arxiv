# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Brain-Like Replay Naturally Emerges in Reinforcement Learning Agents](https://rss.arxiv.org/abs/2402.01467) | 本研究中，我们在使用递归神经网络的强化学习模型中发现了类似大脑回放的现象，并证明其对任务的贡献。这一发现提供了理解回放机制的新视角。 |
| [^2] | [A Neural-Evolutionary Algorithm for Autonomous Transit Network Design](https://arxiv.org/abs/2403.07917) | 提出了一种神经进化算法用于自动公交网络设计，该算法通过训练图神经网络模型作为策略，并将其用作进化算法中的变异操作符，在公交网络设计基准集上优于单独学习策略和简单进化算法方法。 |
| [^3] | [Learning Successor Representations with Distributed Hebbian Temporal Memory.](http://arxiv.org/abs/2310.13391) | 本文提出了一种名为DHTM的算法，它基于因子图形式和多组成神经元模型，利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则来解决在线隐藏表示学习的挑战。实验结果表明，DHTM在变化的环境中比经典的LSTM效果更好，并与更先进的类似RNN的算法性能相当，可以加速继任者表示的时间差异学习。 |

# 详细

[^1]: 强化学习智能体内出现类似大脑回放的现象

    Brain-Like Replay Naturally Emerges in Reinforcement Learning Agents

    [https://rss.arxiv.org/abs/2402.01467](https://rss.arxiv.org/abs/2402.01467)

    本研究中，我们在使用递归神经网络的强化学习模型中发现了类似大脑回放的现象，并证明其对任务的贡献。这一发现提供了理解回放机制的新视角。

    

    大脑区域中普遍观察到的回放现象是否能够在人工智能智能体中自然产生？如果是的话，它是否对任务有所贡献？在本研究中，我们使用基于递归神经网络的强化学习模型，在任务优化的范式下发现了回放的自然出现，模型模拟了海马体和前额叶皮层以及它们之间的相互沟通和感觉皮层的输入。海马体中的回放是由于情景记忆、认知地图以及环境观察而产生的，与动物实验数据相似，并且是高任务性能的有效指标。该模型还成功地重现了局部和非局部的回放，与人类实验数据相符。我们的工作为理解回放机制提供了新的途径。

    Can replay, as a widely observed neural activity pattern in brain regions, particularly in the hippocampus and neocortex, emerge in an artificial agent? If yes, does it contribute to the tasks? In this work, without heavy dependence on complex assumptions, we discover naturally emergent replay under task-optimized paradigm using a recurrent neural network-based reinforcement learning model, which mimics the hippocampus and prefrontal cortex, as well as their intercommunication and the sensory cortex input. The emergent replay in the hippocampus, which results from the episodic memory and cognitive map as well as environment observations, well resembles animal experimental data and serves as an effective indicator of high task performance. The model also successfully reproduces local and nonlocal replay, which matches the human experimental data. Our work provides a new avenue for understanding the mechanisms behind replay.
    
[^2]: 用于自主公交网络设计的神经进化算法

    A Neural-Evolutionary Algorithm for Autonomous Transit Network Design

    [https://arxiv.org/abs/2403.07917](https://arxiv.org/abs/2403.07917)

    提出了一种神经进化算法用于自动公交网络设计，该算法通过训练图神经网络模型作为策略，并将其用作进化算法中的变异操作符，在公交网络设计基准集上优于单独学习策略和简单进化算法方法。

    

    规划公共交通网络是一个具有挑战性的优化问题，但是为了实现自动驾驶公交车的好处是至关重要的。我们提出了一种新颖的算法，用于规划自动驾驶公交车的路线网络。我们首先训练一个图神经网络模型作为构建路线网络的策略，然后将该策略用作进化算法中的多个变异操作符之一。我们在标准的公交网络设计基准集上评估这种算法，并发现它在现实基准实例上的表现比单独学习的策略高出高达20\%，比简单的进化算法方法高出高达53%。

    arXiv:2403.07917v1 Announce Type: cross  Abstract: Planning a public transit network is a challenging optimization problem, but essential in order to realize the benefits of autonomous buses. We propose a novel algorithm for planning networks of routes for autonomous buses. We first train a graph neural net model as a policy for constructing route networks, and then use the policy as one of several mutation operators in a evolutionary algorithm. We evaluate this algorithm on a standard set of benchmarks for transit network design, and find that it outperforms the learned policy alone by up to 20\% and a plain evolutionary algorithm approach by up to 53\% on realistic benchmark instances.
    
[^3]: 使用分布式Hebbian Temporal Memory学习继任者表示法

    Learning Successor Representations with Distributed Hebbian Temporal Memory. (arXiv:2310.13391v1 [cs.LG])

    [http://arxiv.org/abs/2310.13391](http://arxiv.org/abs/2310.13391)

    本文提出了一种名为DHTM的算法，它基于因子图形式和多组成神经元模型，利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则来解决在线隐藏表示学习的挑战。实验结果表明，DHTM在变化的环境中比经典的LSTM效果更好，并与更先进的类似RNN的算法性能相当，可以加速继任者表示的时间差异学习。

    

    本文提出了一种新颖的方法来解决在线隐藏表示学习的挑战，该方法用于在不稳定的、部分可观测的环境中进行决策。所提出的算法，分布式Hebbian Temporal Memory (DHTM)，基于因子图形式和多组成神经元模型。DHTM旨在捕捉顺序数据关系并对未来观察作出累积预测，形成继任者表示。受新皮层的神经生理学模型启发，该算法利用分布式表示、稀疏转移矩阵和局部Hebbian样学习规则克服了传统时间记忆算法（如RNN和HMM）的不稳定性和慢速学习过程。实验结果表明，DHTM优于经典的LSTM，并与更先进的类似RNN的算法性能相当，在变化的环境中加速了继任者表示的时间差异学习。此外，我们还进行了比较。

    This paper presents a novel approach to address the challenge of online hidden representation learning for decision-making under uncertainty in non-stationary, partially observable environments. The proposed algorithm, Distributed Hebbian Temporal Memory (DHTM), is based on factor graph formalism and a multicomponent neuron model. DHTM aims to capture sequential data relationships and make cumulative predictions about future observations, forming Successor Representation (SR). Inspired by neurophysiological models of the neocortex, the algorithm utilizes distributed representations, sparse transition matrices, and local Hebbian-like learning rules to overcome the instability and slow learning process of traditional temporal memory algorithms like RNN and HMM. Experimental results demonstrate that DHTM outperforms classical LSTM and performs comparably to more advanced RNN-like algorithms, speeding up Temporal Difference learning for SR in changing environments. Additionally, we compare
    

