# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction](https://arxiv.org/abs/2402.16174) | GenNBV提出了一种端到端的通用的下一最佳视角策略，通过采用强化学习框架和扩展到5D自由空间的动作空间，实现了无人机从任意视角进行扫描，甚至与未知几何体进行交互的能力，同时提出了多源状态嵌入以增强跨数据集的泛化能力。 |
| [^2] | [KI-PMF: Knowledge Integrated Plausible Motion Forecasting.](http://arxiv.org/abs/2310.12007) | 本研究提出了一种名为KI-PMF的方法，通过结合先验知识，对交通参与者的未来行动进行准确预测，遵循车辆的运动约束和行驶环境的几何形状。通过条件化网络以遵循物理定律，可以获得准确和安全的预测，对于在实际环境中维护自动驾驶汽车的安全和效率至关重要。 |

# 详细

[^1]: GenNBV: 通用的主动式三维重建下一最佳视角策略

    GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction

    [https://arxiv.org/abs/2402.16174](https://arxiv.org/abs/2402.16174)

    GenNBV提出了一种端到端的通用的下一最佳视角策略，通过采用强化学习框架和扩展到5D自由空间的动作空间，实现了无人机从任意视角进行扫描，甚至与未知几何体进行交互的能力，同时提出了多源状态嵌入以增强跨数据集的泛化能力。

    

    最近的神经辐射场的技术进步实现了大规模场景的真实数字化, 但是图像捕获过程仍然耗时且劳动密集。先前的研究尝试使用主动式三维重建的下一最佳视角（NBV）策略来自动化这一过程。然而，现有的NBV策略严重依赖手工设计的标准、有限的动作空间，或者是针对每个场景优化的表示。这些约束限制了它们在跨数据集中的泛化能力。为了克服这些问题，我们提出了GenNBV，一个端到端通用的NBV策略。我们的策略采用基于强化学习（RL）的框架，将典型有限的动作空间扩展到5D自由空间。它赋予了我们的代理机无人机在训练过程中可以从任何视角进行扫描，甚至可以与未见几何体进行交互。为了增强跨数据集的泛化能力，我们还提出了一种新颖的多源状态嵌入，包括几何、语义和动作表示。

    arXiv:2402.16174v1 Announce Type: cross  Abstract: While recent advances in neural radiance field enable realistic digitization for large-scale scenes, the image-capturing process is still time-consuming and labor-intensive. Previous works attempt to automate this process using the Next-Best-View (NBV) policy for active 3D reconstruction. However, the existing NBV policies heavily rely on hand-crafted criteria, limited action space, or per-scene optimized representations. These constraints limit their cross-dataset generalizability. To overcome them, we propose GenNBV, an end-to-end generalizable NBV policy. Our policy adopts a reinforcement learning (RL)-based framework and extends typical limited action space to 5D free space. It empowers our agent drone to scan from any viewpoint, and even interact with unseen geometries during training. To boost the cross-dataset generalizability, we also propose a novel multi-source state embedding, including geometric, semantic, and action repres
    
[^2]: KI-PMF：知识综合的合理动作预测

    KI-PMF: Knowledge Integrated Plausible Motion Forecasting. (arXiv:2310.12007v1 [cs.RO])

    [http://arxiv.org/abs/2310.12007](http://arxiv.org/abs/2310.12007)

    本研究提出了一种名为KI-PMF的方法，通过结合先验知识，对交通参与者的未来行动进行准确预测，遵循车辆的运动约束和行驶环境的几何形状。通过条件化网络以遵循物理定律，可以获得准确和安全的预测，对于在实际环境中维护自动驾驶汽车的安全和效率至关重要。

    

    准确预测交通参与者的行动对大规模部署自动驾驶汽车至关重要。当前的轨迹预测方法主要集中在优化特定度量的损失函数上，这可能导致预测不符合物理定律或违反外部约束条件。我们的目标是结合明确的先验知识，使网络能够预测未来轨迹，符合车辆的运动约束和行驶环境的几何形状。为了实现这一目标，我们引入了非参数剪枝层和注意力层来整合定义的先验知识。我们的方法旨在确保交通参与者在复杂和动态情况下的到达可达性保证。通过将网络条件化为遵循物理定律，我们可以获得准确和安全的预测，这对于在实际世界环境中维护自动驾驶汽车的安全和效率至关重要。

    Accurately forecasting the motion of traffic actors is crucial for the deployment of autonomous vehicles at a large scale. Current trajectory forecasting approaches primarily concentrate on optimizing a loss function with a specific metric, which can result in predictions that do not adhere to physical laws or violate external constraints. Our objective is to incorporate explicit knowledge priors that allow a network to forecast future trajectories in compliance with both the kinematic constraints of a vehicle and the geometry of the driving environment. To achieve this, we introduce a non-parametric pruning layer and attention layers to integrate the defined knowledge priors. Our proposed method is designed to ensure reachability guarantees for traffic actors in both complex and dynamic situations. By conditioning the network to follow physical laws, we can obtain accurate and safe predictions, essential for maintaining autonomous vehicles' safety and efficiency in real-world settings
    

