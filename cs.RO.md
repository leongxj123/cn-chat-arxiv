# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning](https://arxiv.org/abs/2403.12856) | 本文提出了一种无需专门神经网络组件的等变策略和不变值函数构建方法，在基于地图的路径规划中展示了等变集合和正则化如何提高样本效率和性能 |
| [^2] | [3D Diffuser Actor: Policy Diffusion with 3D Scene Representations](https://arxiv.org/abs/2402.10885) | 通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。 |

# 详细

[^1]: 基于地图的路径规划中的等变集合和正则化的强化学习

    Equivariant Ensembles and Regularization for Reinforcement Learning in Map-based Path Planning

    [https://arxiv.org/abs/2403.12856](https://arxiv.org/abs/2403.12856)

    本文提出了一种无需专门神经网络组件的等变策略和不变值函数构建方法，在基于地图的路径规划中展示了等变集合和正则化如何提高样本效率和性能

    

    在强化学习（RL）中，利用环境的对称性可以显著增强效率、鲁棒性和性能。然而，确保深度RL策略和值网络分别是等变和不变的以利用这些对称性是一个重大挑战。相关工作尝试通过构造具有等变性和不变性的网络来设计，这限制了它们只能使用非常受限的组件库，进而阻碍了网络的表现能力。本文提出了一种构建等变策略和不变值函数的方法，而无需专门的神经网络组件，我们将其称为等变集合。我们进一步添加了一个正则化项，用于在训练过程中增加归纳偏差。在基于地图的路径规划案例研究中，我们展示了等变集合和正则化如何有益于样本效率和性能。

    arXiv:2403.12856v1 Announce Type: new  Abstract: In reinforcement learning (RL), exploiting environmental symmetries can significantly enhance efficiency, robustness, and performance. However, ensuring that the deep RL policy and value networks are respectively equivariant and invariant to exploit these symmetries is a substantial challenge. Related works try to design networks that are equivariant and invariant by construction, limiting them to a very restricted library of components, which in turn hampers the expressiveness of the networks. This paper proposes a method to construct equivariant policies and invariant value functions without specialized neural network components, which we term equivariant ensembles. We further add a regularization term for adding inductive bias during training. In a map-based path planning case study, we show how equivariant ensembles and regularization benefit sample efficiency and performance.
    
[^2]: 基于3D场景表示的3D扩散器Actor：通过策略扩散进行机器人操作

    3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

    [https://arxiv.org/abs/2402.10885](https://arxiv.org/abs/2402.10885)

    通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。

    

    我们将扩散策略和3D场景表示相结合，用于机器人操作。扩散策略通过条件扩散模型学习基于机器人和环境状态的动作分布。最近，它们已经表现出优于确定性和其他基于状态的动作分布学习方法。3D机器人策略使用从单个或多个摄像头视角获取的感应深度聚合的3D场景特征表示。它们已经证明比其2D对应物在摄像机视角上具有更好的泛化能力。我们统一了这两条线路的工作，并提出了3D扩散器Actor，这是一个神经策略架构，它在给定语言指令的情况下，构建视觉场景的3D表示，并在其上进行条件迭代去噪机器人末端执行器的3D旋转和平移。在每个去噪迭代中，我们的模型将末端执行器姿态估计表示为3D场景令牌，并预测t

    arXiv:2402.10885v1 Announce Type: cross  Abstract: We marry diffusion policies and 3D scene representations for robot manipulation. Diffusion policies learn the action distribution conditioned on the robot and environment state using conditional diffusion models. They have recently shown to outperform both deterministic and alternative state-conditioned action distribution learning methods. 3D robot policies use 3D scene feature representations aggregated from a single or multiple camera views using sensed depth. They have shown to generalize better than their 2D counterparts across camera viewpoints. We unify these two lines of work and present 3D Diffuser Actor, a neural policy architecture that, given a language instruction, builds a 3D representation of the visual scene and conditions on it to iteratively denoise 3D rotations and translations for the robot's end-effector. At each denoising iteration, our model represents end-effector pose estimates as 3D scene tokens and predicts t
    

