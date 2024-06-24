# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay](https://arxiv.org/abs/2403.11852) | 本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。 |
| [^2] | [Hierarchical hybrid modeling for flexible tool use](https://arxiv.org/abs/2402.10088) | 本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。 |
| [^3] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |
| [^4] | [Diverse Offline Imitation via Fenchel Duality.](http://arxiv.org/abs/2307.11373) | 本文提出了一个离线技能发现算法，通过Fenchel对偶方法将强化学习和无监督技能发现结合起来，实现学习与专家相一致的多样的技能。 |
| [^5] | [RoMo-HER: Robust Model-based Hindsight Experience Replay.](http://arxiv.org/abs/2306.16061) | RoMo-HER是一个鲁棒的基于模型的事后经验回放方法，通过使用机器人操作环境中的动力学模型和前瞻重新标记技术，提高了样本利用效率。 |
| [^6] | [Hierarchical Path-planning from Speech Instructions with Spatial Concept-based Topometric Semantic Mapping.](http://arxiv.org/abs/2203.10820) | 该论文提出了一种基于拓扑语义映射的分层路径规划方法，可以通过语音指令和waypoint灵活地规划路径，包括地点连通性，提供了一种快速近似推理方法。 |

# 详细

[^1]: 具有潜在状态推断的强化学习在自动匝道合并中的应用

    Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay

    [https://arxiv.org/abs/2403.11852](https://arxiv.org/abs/2403.11852)

    本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。

    

    本文提出了一种解决自动匝道合并问题的新方法，其中自动驾驶车辆需要无缝地融入多车道高速公路上的车流。我们介绍了Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS)代理，旨在在没有关于周围车辆意图或驾驶风格的全面知识的情况下安全执行匝道合并任务。我们还提出了该代理的增强版AL3IS，考虑了观测延迟，使代理能够在具有车辆间通信延迟的现实环境中做出更稳健的决策。通过通过潜在状态建模环境中的不可观察方面，如其他驾驶员的意图，我们的方法增强了代理适应动态交通状况、优化合并操作并确保与其他车辆进行安全互动的能力。

    arXiv:2403.11852v1 Announce Type: cross  Abstract: This paper presents a novel approach to address the challenging problem of autonomous on-ramp merging, where a self-driving vehicle needs to seamlessly integrate into a flow of vehicles on a multi-lane highway. We introduce the Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS) agent, designed to perform the on-ramp merging task safely without comprehensive knowledge about surrounding vehicles' intents or driving styles. We also present an augmentation of this agent called AL3IS that accounts for observation delays, allowing the agent to make more robust decisions in real-world environments with vehicle-to-vehicle (V2V) communication delays. By modeling the unobservable aspects of the environment through latent states, such as other drivers' intents, our approach enhances the agent's ability to adapt to dynamic traffic conditions, optimize merging maneuvers, and ensure safe interactions with other vehi
    
[^2]: 分层混合建模用于灵活工具使用

    Hierarchical hybrid modeling for flexible tool use

    [https://arxiv.org/abs/2402.10088](https://arxiv.org/abs/2402.10088)

    本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。

    

    在最近提出的主动推理计算框架中，离散模型可以与连续模型相结合，以在不断变化的环境中进行决策。从另一个角度来看，简单的代理可以组合在一起，以更好地捕捉世界的因果关系。我们如何将这两个特点结合起来实现高效的目标导向行为？我们提出了一个架构，由多个混合 - 连续和离散 - 单元组成，复制代理的配置，由高级离散模型控制，实现动态规划和同步行为。每个层次内部的进一步分解可以以分层方式表示与self相关的其他代理和对象。我们在一个非平凡的任务上评估了这种分层混合模型：在拾取一个移动工具后到达一个移动物体。这项研究扩展了以推理为控制的先前工作，并提出了一种替代方案。

    arXiv:2402.10088v1 Announce Type: cross  Abstract: In a recent computational framework called active inference, discrete models can be linked to their continuous counterparts to perform decision-making in changing environments. From another perspective, simple agents can be combined to better capture the causal relationships of the world. How can we use these two features together to achieve efficient goal-directed behavior? We present an architecture composed of several hybrid -- continuous and discrete -- units replicating the agent's configuration, controlled by a high-level discrete model that achieves dynamic planning and synchronized behavior. Additional factorizations within each level allow to represent hierarchically other agents and objects in relation to the self. We evaluate this hierarchical hybrid model on a non-trivial task: reaching a moving object after having picked a moving tool. This study extends past work on control as inference and proposes an alternative directi
    
[^3]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    
[^4]: 通过Fenchel对偶实现多样的离线模仿

    Diverse Offline Imitation via Fenchel Duality. (arXiv:2307.11373v1 [cs.LG])

    [http://arxiv.org/abs/2307.11373](http://arxiv.org/abs/2307.11373)

    本文提出了一个离线技能发现算法，通过Fenchel对偶方法将强化学习和无监督技能发现结合起来，实现学习与专家相一致的多样的技能。

    

    在无监督技能发现领域，最近取得了显著进展，各种工作提出了以互信息为基础的目标，作为内在驱动。先前的工作主要集中在设计需要在线环境访问的算法。相比之下，我们开发了一个\textit{离线}技能发现算法。我们的问题形式化考虑了在KL-散度约束下最大化互信息目标。更确切地说，约束确保每个技能的状态占用保持在一个具有良好状态操作覆盖率的离线数据集的支持范围内与专家的状态占用逼近。我们的主要贡献是连接Fenchel对偶、强化学习和无监督技能发现，并给出一个简单的离线算法，用于学习与专家相一致的多样的技能。

    There has been significant recent progress in the area of unsupervised skill discovery, with various works proposing mutual information based objectives, as a source of intrinsic motivation. Prior works predominantly focused on designing algorithms that require online access to the environment. In contrast, we develop an \textit{offline} skill discovery algorithm. Our problem formulation considers the maximization of a mutual information objective constrained by a KL-divergence. More precisely, the constraints ensure that the state occupancy of each skill remains close to the state occupancy of an expert, within the support of an offline dataset with good state-action coverage. Our main contribution is to connect Fenchel duality, reinforcement learning and unsupervised skill discovery, and to give a simple offline algorithm for learning diverse skills that are aligned with an expert.
    
[^5]: RoMo-HER: 鲁棒的基于模型的事后经验回放方法

    RoMo-HER: Robust Model-based Hindsight Experience Replay. (arXiv:2306.16061v1 [cs.RO])

    [http://arxiv.org/abs/2306.16061](http://arxiv.org/abs/2306.16061)

    RoMo-HER是一个鲁棒的基于模型的事后经验回放方法，通过使用机器人操作环境中的动力学模型和前瞻重新标记技术，提高了样本利用效率。

    

    在多目标强化学习中，稀疏奖励是导致样本利用效率低的因素之一。基于事后经验回放（HER），已经提出了基于模型的重新标记方法，通过与训练模型进行交互获取虚拟轨迹来重新标记目标，在准确可建模的稀疏奖励环境中能够有效增强样本利用效率。然而，在机器人操作环境中，它们是无效的。在我们的论文中，我们设计了一个称为RoMo-HER的鲁棒框架，它可以有效地利用机器人操作环境中的动力学模型来提高样本利用效率。RoMo-HER基于动力学模型和一种称为前瞻重新标记（FR）的新型目标重新标记技术构建，该技术通过特定策略选择预测起始状态，预测起始状态的未来轨迹，然后使用动力学模型和最新的信息重新标记目标。

    Sparse rewards are one of the factors leading to low sample efficiency in multi-goal reinforcement learning (RL). Based on Hindsight Experience Replay (HER), model-based relabeling methods have been proposed to relabel goals using virtual trajectories obtained by interacting with the trained model, which can effectively enhance the sample efficiency in accurately modelable sparse-reward environments. However, they are ineffective in robot manipulation environment. In our paper, we design a robust framework called Robust Model-based Hindsight Experience Replay (RoMo-HER) which can effectively utilize the dynamical model in robot manipulation environments to enhance the sample efficiency. RoMo-HER is built upon a dynamics model and a novel goal relabeling technique called Foresight relabeling (FR), which selects the prediction starting state with a specific strategy, predicts the future trajectory of the starting state, and then relabels the goal using the dynamics model and the latest p
    
[^6]: 利用基于空间概念的拓扑语义映射进行从语音指令的分层路径规划

    Hierarchical Path-planning from Speech Instructions with Spatial Concept-based Topometric Semantic Mapping. (arXiv:2203.10820v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2203.10820](http://arxiv.org/abs/2203.10820)

    该论文提出了一种基于拓扑语义映射的分层路径规划方法，可以通过语音指令和waypoint灵活地规划路径，包括地点连通性，提供了一种快速近似推理方法。

    

    在现实世界中，使用人类语音指令进行机器人导航至目的地是自主移动机器人至关重要的。然而，机器人可以采用不同的路径到达同一目标，而最短路径不一定是最优的。因此需要一种方法来灵活地接受waypoint的指标，规划更好的替代路径，即使需要绕路。此外，机器人需要实时推理能力。本研究旨在通过拓扑语义映射实现一个分层的空间表示，并结合语音指令和waypoint进行路径规划。我们提出了SpCoTMHP，一种利用多模式空间概念的层次路径规划方法，包括地点连通性。这种方法提供了一种新颖的集成概率生成模型和快速近似推理方法，层次结构中的各个级别之间可以相互交互。

    Navigating to destinations using human speech instructions is essential for autonomous mobile robots operating in the real world. Although robots can take different paths toward the same goal, the shortest path is not always optimal. A desired approach is to flexibly accommodate waypoint specifications, planning a better alternative path, even with detours. Furthermore, robots require real-time inference capabilities. Spatial representations include semantic, topological, and metric levels, each capturing different aspects of the environment. This study aims to realize a hierarchical spatial representation by a topometric semantic map and path planning with speech instructions, including waypoints. We propose SpCoTMHP, a hierarchical path-planning method that utilizes multimodal spatial concepts, incorporating place connectivity. This approach provides a novel integrated probabilistic generative model and fast approximate inference, with interaction among the hierarchy levels. A formul
    

