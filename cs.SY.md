# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Brain-Like Replay Naturally Emerges in Reinforcement Learning Agents](https://rss.arxiv.org/abs/2402.01467) | 本研究中，我们在使用递归神经网络的强化学习模型中发现了类似大脑回放的现象，并证明其对任务的贡献。这一发现提供了理解回放机制的新视角。 |
| [^2] | [A Moreau Envelope Approach for LQR Meta-Policy Estimation](https://arxiv.org/abs/2403.17364) | 提出了一种基于Moreau包络的替代LQR成本，可有效调整到新实现的元策略，并设计了找到近似一阶稳定点的算法。 |
| [^3] | [An active learning method for solving competitive multi-agent decision-making and control problems.](http://arxiv.org/abs/2212.12561) | 我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。 |

# 详细

[^1]: 强化学习智能体内出现类似大脑回放的现象

    Brain-Like Replay Naturally Emerges in Reinforcement Learning Agents

    [https://rss.arxiv.org/abs/2402.01467](https://rss.arxiv.org/abs/2402.01467)

    本研究中，我们在使用递归神经网络的强化学习模型中发现了类似大脑回放的现象，并证明其对任务的贡献。这一发现提供了理解回放机制的新视角。

    

    大脑区域中普遍观察到的回放现象是否能够在人工智能智能体中自然产生？如果是的话，它是否对任务有所贡献？在本研究中，我们使用基于递归神经网络的强化学习模型，在任务优化的范式下发现了回放的自然出现，模型模拟了海马体和前额叶皮层以及它们之间的相互沟通和感觉皮层的输入。海马体中的回放是由于情景记忆、认知地图以及环境观察而产生的，与动物实验数据相似，并且是高任务性能的有效指标。该模型还成功地重现了局部和非局部的回放，与人类实验数据相符。我们的工作为理解回放机制提供了新的途径。

    Can replay, as a widely observed neural activity pattern in brain regions, particularly in the hippocampus and neocortex, emerge in an artificial agent? If yes, does it contribute to the tasks? In this work, without heavy dependence on complex assumptions, we discover naturally emergent replay under task-optimized paradigm using a recurrent neural network-based reinforcement learning model, which mimics the hippocampus and prefrontal cortex, as well as their intercommunication and the sensory cortex input. The emergent replay in the hippocampus, which results from the episodic memory and cognitive map as well as environment observations, well resembles animal experimental data and serves as an effective indicator of high task performance. The model also successfully reproduces local and nonlocal replay, which matches the human experimental data. Our work provides a new avenue for understanding the mechanisms behind replay.
    
[^2]: 一种适用于LQR元策略估计的Moreau包络方法

    A Moreau Envelope Approach for LQR Meta-Policy Estimation

    [https://arxiv.org/abs/2403.17364](https://arxiv.org/abs/2403.17364)

    提出了一种基于Moreau包络的替代LQR成本，可有效调整到新实现的元策略，并设计了找到近似一阶稳定点的算法。

    

    我们研究了在线性时不变离散时间不确定动态系统中的线性二次型调节器（LQR）策略估计问题。我们提出了一种基于Moreau包络的替代LQR成本，由不确定系统的有限实现构建，以定义一个对新实现有效调整的元策略。此外，我们设计了一个算法来找到元LQR成本函数的近似一阶稳定点。数值结果表明，所提出的方法在新实现的线性系统上胜过了控制器的朴素平均。我们还提供了实证证据表明，我们的方法比模型不可知元学习（MAML）方法具有更好的样本复杂性。

    arXiv:2403.17364v1 Announce Type: cross  Abstract: We study the problem of policy estimation for the Linear Quadratic Regulator (LQR) in discrete-time linear time-invariant uncertain dynamical systems. We propose a Moreau Envelope-based surrogate LQR cost, built from a finite set of realizations of the uncertain system, to define a meta-policy efficiently adjustable to new realizations. Moreover, we design an algorithm to find an approximate first-order stationary point of the meta-LQR cost function. Numerical results show that the proposed approach outperforms naive averaging of controllers on new realizations of the linear system. We also provide empirical evidence that our method has better sample complexity than Model-Agnostic Meta-Learning (MAML) approaches.
    
[^3]: 解决竞争性多智能体决策和控制问题的主动学习方法

    An active learning method for solving competitive multi-agent decision-making and control problems. (arXiv:2212.12561v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2212.12561](http://arxiv.org/abs/2212.12561)

    我们提出了一个基于主动学习的方法，用于解决竞争性多智能体决策和控制问题。通过重构私有策略和预测稳态行动配置文件，外部观察者可以成功进行预测和优化策略。

    

    我们提出了一种基于主动学习的方案，用于重构由相互作用代理人群体执行的私有策略，并预测底层多智能体交互过程的确切结果，这里被认为是一个稳定的行动配置文件。我们设想了一个场景，在这个场景中，一个具有学习程序的外部观察者可以通过私有的行动-反应映射进行查询和观察代理人的反应，集体的不动点对应于一个稳态配置文件。通过迭代地收集有意义的数据和更新行动-反应映射的参数估计，我们建立了评估所提出的主动学习方法的渐近性质的充分条件，以便如果收敛发生，它只能朝向一个稳态行动配置文件。这一事实导致了两个主要结果：i）学习局部精确的行动-反应映射替代物使得外部观察者能够成功完成其预测任务，ii）与代理人的互动提供了一种方法来优化策略以达到最佳效果。

    We propose a scheme based on active learning to reconstruct private strategies executed by a population of interacting agents and predict an exact outcome of the underlying multi-agent interaction process, here identified as a stationary action profile. We envision a scenario where an external observer, endowed with a learning procedure, can make queries and observe the agents' reactions through private action-reaction mappings, whose collective fixed point corresponds to a stationary profile. By iteratively collecting sensible data and updating parametric estimates of the action-reaction mappings, we establish sufficient conditions to assess the asymptotic properties of the proposed active learning methodology so that, if convergence happens, it can only be towards a stationary action profile. This fact yields two main consequences: i) learning locally-exact surrogates of the action-reaction mappings allows the external observer to succeed in its prediction task, and ii) working with 
    

