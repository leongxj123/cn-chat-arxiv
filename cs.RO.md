# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Multi-Agent Mapping for Planetary Exploration](https://arxiv.org/abs/2404.02289) | 联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。 |
| [^2] | [ContactHandover: Contact-Guided Robot-to-Human Object Handover](https://arxiv.org/abs/2404.01402) | ContactHandover是一个机器人向人类递送物体的系统，通过接触引导的抓取和物体递送阶段来实现成功的物体递送。 |
| [^3] | [RPMArt: Towards Robust Perception and Manipulation for Articulated Objects](https://arxiv.org/abs/2403.16023) | 提出了面向关节对象的健壮感知和操作框架RPMArt，主要贡献是能够稳健地预测关节参数和可信点的RoArtNet。 |
| [^4] | [STAMP: Differentiable Task and Motion Planning via Stein Variational Gradient Descent.](http://arxiv.org/abs/2310.01775) | STAMP是一种基于Stein变分梯度下降的算法，通过并行化和可微仿真高效地搜索多个多样化的任务和运动规划解决方案。 |
| [^5] | [RL + Model-based Control: Using On-demand Optimal Control to Learn Versatile Legged Locomotion.](http://arxiv.org/abs/2305.17842) | 本文提出了一种 RL+模型控制框架以开发出可以有效可靠地学习的健壮控制策略，通过整合有限时间最优控制生成的按需参考运动分散 RL 过程，同时克服了建模简化的固有局限性，在足式 locomotion 上实现了多功能和强健，能泛化参考运动并处理更复杂的运动任务。 |

# 详细

[^1]: 行星探测的联邦多智能体建图

    Federated Multi-Agent Mapping for Planetary Exploration

    [https://arxiv.org/abs/2404.02289](https://arxiv.org/abs/2404.02289)

    联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。

    

    在多智能体机器人探测中，管理和有效利用动态环境产生的大量异构数据构成了一个重要挑战。联邦学习（FL）是一种有前途的分布式映射方法，它解决了协作学习中去中心化数据的挑战。FL使多个智能体之间可以进行联合模型训练，而无需集中化或共享原始数据，克服了带宽和存储限制。我们的方法利用隐式神经映射，将地图表示为由神经网络学习的连续函数，以便实现紧凑和适应性的表示。我们进一步通过在地球数据集上进行元初始化来增强这一方法，预训练网络以快速学习新的地图结构。这种组合在诸如火星地形和冰川等不同领域展现了较强的泛化能力。我们对这一方法进行了严格评估，展示了其有效性。

    arXiv:2404.02289v1 Announce Type: cross  Abstract: In multi-agent robotic exploration, managing and effectively utilizing the vast, heterogeneous data generated from dynamic environments poses a significant challenge. Federated learning (FL) is a promising approach for distributed mapping, addressing the challenges of decentralized data in collaborative learning. FL enables joint model training across multiple agents without requiring the centralization or sharing of raw data, overcoming bandwidth and storage constraints. Our approach leverages implicit neural mapping, representing maps as continuous functions learned by neural networks, for compact and adaptable representations. We further enhance this approach with meta-initialization on Earth datasets, pre-training the network to quickly learn new map structures. This combination demonstrates strong generalization to diverse domains like Martian terrain and glaciers. We rigorously evaluate this approach, demonstrating its effectiven
    
[^2]: ContactHandover: 接触引导的机器人向人类递送物体

    ContactHandover: Contact-Guided Robot-to-Human Object Handover

    [https://arxiv.org/abs/2404.01402](https://arxiv.org/abs/2404.01402)

    ContactHandover是一个机器人向人类递送物体的系统，通过接触引导的抓取和物体递送阶段来实现成功的物体递送。

    

    机器人向人类递送物体是许多人机协作任务中的重要一步。成功的递送需要机器人保持对物体的稳定抓取，同时确保人类以一种自然且易于使用的方式接收物体。我们提出了ContactHandover，这是一个机器人向人类递送物体的系统，包括两个阶段：接触引导的抓取阶段和物体递送阶段。在抓取阶段，ContactHandover预测机器人的6自由度抓取姿势和人类接触点在物体上的3D可供性图。机器人的抓取姿势通过惩罚那些阻碍人类接触点的姿势进行重新排序，并执行排名最高的抓取。在递送阶段，通过最大化靠近人类的接触点并最小化人类手臂关节扭矩和位移来计算机器人末端执行器姿势。我们在27种不同家用物品上评估了我们的系统，并展示了o

    arXiv:2404.01402v1 Announce Type: cross  Abstract: Robot-to-human object handover is an important step in many human robot collaboration tasks. A successful handover requires the robot to maintain a stable grasp on the object while making sure the human receives the object in a natural and easy-to-use manner. We propose ContactHandover, a robot to human handover system that consists of two phases: a contact-guided grasping phase and an object delivery phase. During the grasping phase, ContactHandover predicts both 6-DoF robot grasp poses and a 3D affordance map of human contact points on the object. The robot grasp poses are reranked by penalizing those that block human contact points, and the robot executes the highest ranking grasp. During the delivery phase, the robot end effector pose is computed by maximizing human contact points close to the human while minimizing the human arm joint torques and displacements. We evaluate our system on 27 diverse household objects and show that o
    
[^3]: RPMArt：面向关节对象的健壮感知和操作

    RPMArt: Towards Robust Perception and Manipulation for Articulated Objects

    [https://arxiv.org/abs/2403.16023](https://arxiv.org/abs/2403.16023)

    提出了面向关节对象的健壮感知和操作框架RPMArt，主要贡献是能够稳健地预测关节参数和可信点的RoArtNet。

    

    关节对象在日常生活中很常见。对于真实世界的机器人应用来说，机器人能够表现出对关节对象的健壮感知和操作技能是至关重要的。然而，现有的关节对象方法不够解决点云中的噪声问题，难以弥合模拟与现实之间的差距，从而限制了在真实场景中的实际部署。为了解决这些挑战，我们提出了一个面向关节对象的健壮感知和操作的框架（RPMArt），该框架学习如何从嘈杂的点云中估计关节参数并操作关节部分。我们的主要贡献是一个健壮关节网络（RoArtNet），通过局部特征学习和点元组投票能够稳健地预测关节参数和可信点。此外，我们引入了一个关节感知分类方案来增强其能力。

    arXiv:2403.16023v1 Announce Type: cross  Abstract: Articulated objects are commonly found in daily life. It is essential that robots can exhibit robust perception and manipulation skills for articulated objects in real-world robotic applications. However, existing methods for articulated objects insufficiently address noise in point clouds and struggle to bridge the gap between simulation and reality, thus limiting the practical deployment in real-world scenarios. To tackle these challenges, we propose a framework towards Robust Perception and Manipulation for Articulated Objects (RPMArt), which learns to estimate the articulation parameters and manipulate the articulation part from the noisy point cloud. Our primary contribution is a Robust Articulation Network (RoArtNet) that is able to predict both joint parameters and affordable points robustly by local feature learning and point tuple voting. Moreover, we introduce an articulation-aware classification scheme to enhance its ability
    
[^4]: STAMP：通过Stein变分梯度下降实现可微的任务和运动规划

    STAMP: Differentiable Task and Motion Planning via Stein Variational Gradient Descent. (arXiv:2310.01775v1 [cs.RO])

    [http://arxiv.org/abs/2310.01775](http://arxiv.org/abs/2310.01775)

    STAMP是一种基于Stein变分梯度下降的算法，通过并行化和可微仿真高效地搜索多个多样化的任务和运动规划解决方案。

    

    许多操作任务，如使用工具或装配零件，往往需要符号和几何推理。任务和运动规划（TAMP）算法通常通过对高级任务序列进行树搜索并检查运动学和动力学可行性来解决这些问题。虽然性能良好，但大多数现有算法的效率非常低，因为其时间复杂性随可能动作和物体数量的增加呈指数增长。此外，它们只能找到单个解决方案，而可能存在许多可行的计划。为了解决这些限制，我们提出了一种名为Stein任务和运动规划（STAMP）的新算法，它利用并行化和可微仿真来高效地搜索多个多样化的计划。STAMP将离散和连续的TAMP问题转化为可以使用变分推断解决的连续优化问题。我们的算法基于Stein变分梯度下降，一种概率推断方法。

    Planning for many manipulation tasks, such as using tools or assembling parts, often requires both symbolic and geometric reasoning. Task and Motion Planning (TAMP) algorithms typically solve these problems by conducting a tree search over high-level task sequences while checking for kinematic and dynamic feasibility. While performant, most existing algorithms are highly inefficient as their time complexity grows exponentially with the number of possible actions and objects. Additionally, they only find a single solution to problems in which many feasible plans may exist. To address these limitations, we propose a novel algorithm called Stein Task and Motion Planning (STAMP) that leverages parallelization and differentiable simulation to efficiently search for multiple diverse plans. STAMP relaxes discrete-and-continuous TAMP problems into continuous optimization problems that can be solved using variational inference. Our algorithm builds upon Stein Variational Gradient Descent, a gra
    
[^5]: RL+模型控制：使用按需最优控制学习多功能足式 locomotion

    RL + Model-based Control: Using On-demand Optimal Control to Learn Versatile Legged Locomotion. (arXiv:2305.17842v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2305.17842](http://arxiv.org/abs/2305.17842)

    本文提出了一种 RL+模型控制框架以开发出可以有效可靠地学习的健壮控制策略，通过整合有限时间最优控制生成的按需参考运动分散 RL 过程，同时克服了建模简化的固有局限性，在足式 locomotion 上实现了多功能和强健，能泛化参考运动并处理更复杂的运动任务。

    

    本文提出了一种控制框架，将基于模型的最优控制和强化学习（RL）相结合，实现了多功能和强健的足式 locomotion。我们的方法通过整合有限时间最优控制生成的按需参考运动来增强 RL 训练过程，覆盖了广泛的速度和步态。这些参考运动作为 RL 策略模仿的目标，导致开发出可有效可靠地学习的健壮控制策略。此外，通过考虑全身动力学，RL 克服了建模简化的固有局限性。通过仿真和硬件实验，我们展示了 RL 训练过程在我们的框架内的强健性和可控性。此外，我们的方法展示了泛化参考运动和处理可能对简化模型构成挑战的更复杂的运动任务的能力，利用了 RL 的灵活性。

    This letter presents a control framework that combines model-based optimal control and reinforcement learning (RL) to achieve versatile and robust legged locomotion. Our approach enhances the RL training process by incorporating on-demand reference motions generated through finite-horizon optimal control, covering a broad range of velocities and gaits. These reference motions serve as targets for the RL policy to imitate, resulting in the development of robust control policies that can be learned efficiently and reliably. Moreover, by considering whole-body dynamics, RL overcomes the inherent limitations of modelling simplifications. Through simulation and hardware experiments, we demonstrate the robustness and controllability of the RL training process within our framework. Furthermore, our method demonstrates the ability to generalize reference motions and handle more complex locomotion tasks that may pose challenges for the simplified model, leveraging the flexibility of RL.
    

