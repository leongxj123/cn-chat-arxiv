# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning](https://arxiv.org/abs/2403.09793) | 提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。 |
| [^2] | [CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks](https://arxiv.org/abs/2402.17363) | CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题 |
| [^3] | [SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes](https://arxiv.org/abs/2312.03297) | SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。 |
| [^4] | [Learning to Visually Connect Actions and their Effects.](http://arxiv.org/abs/2401.10805) | 该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。 |

# 详细

[^1]: 社会整合导航：具有深度强化学习的社交行动机器人

    Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.09793](https://arxiv.org/abs/2403.09793)

    提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。

    

    移动机器人正在广泛应用于各种拥挤场景，并成为我们社会的一部分。一个具有个体人类考虑的社会可接受的导航行为对于可扩展的应用和人类接受至关重要。最近使用深度强化学习（DRL）方法来学习机器人的导航策略，并对机器人与人类之间的复杂交互进行建模。我们建议根据机器人展示的社交行为将现有基于DRL的导航方法分为具有缺乏社交行为的社会碰撞回避和具有明确预定义社交行为的社会意识方法。此外，我们提出了一种新颖的社会整合导航方法，其中机器人的社交行为是自适应的，并且是通过与人类的互动而产生的。我们的方法的构式源自社会学定义，

    arXiv:2403.09793v1 Announce Type: cross  Abstract: Mobile robots are being used on a large scale in various crowded situations and become part of our society. The socially acceptable navigation behavior of a mobile robot with individual human consideration is an essential requirement for scalable applications and human acceptance. Deep Reinforcement Learning (DRL) approaches are recently used to learn a robot's navigation policy and to model the complex interactions between robots and humans. We propose to divide existing DRL-based navigation approaches based on the robot's exhibited social behavior and distinguish between social collision avoidance with a lack of social behavior and socially aware approaches with explicit predefined social behavior. In addition, we propose a novel socially integrated navigation approach where the robot's social behavior is adaptive and emerges from the interaction with humans. The formulation of our approach is derived from a sociological definition, 
    
[^2]: CGGM：一种具有自适应稀疏性的条件图生成模型，用于物联网网络中节点异常检测

    CGGM: A conditional graph generation model with adaptive sparsity for node anomaly detection in IoT networks

    [https://arxiv.org/abs/2402.17363](https://arxiv.org/abs/2402.17363)

    CGGM是一种新颖的图生成模型，通过自适应稀疏性生成邻接矩阵，解决了物联网网络中节点异常检测中节点类别不平衡的问题

    

    动态图被广泛用于检测物联网中节点的异常行为。生成模型通常用于解决动态图中节点类别不平衡的问题。然而，它面临的约束包括邻接关系的单调性，为节点构建多维特征的困难，以及缺乏端到端生成多类节点的方法。本文提出了一种名为CGGM的新颖图生成模型，专门设计用于生成少数类别中更多节点。通过自适应稀疏性生成邻接矩阵的机制增强了其结构的灵活性。特征生成模块名为多维特征生成器（MFG），可生成包括拓扑信息在内的节点特征。标签被转换为嵌入向量，用作条件。

    arXiv:2402.17363v1 Announce Type: cross  Abstract: Dynamic graphs are extensively employed for detecting anomalous behavior in nodes within the Internet of Things (IoT). Generative models are often used to address the issue of imbalanced node categories in dynamic graphs. Nevertheless, the constraints it faces include the monotonicity of adjacency relationships, the difficulty in constructing multi-dimensional features for nodes, and the lack of a method for end-to-end generation of multiple categories of nodes. This paper presents a novel graph generation model, called CGGM, designed specifically to generate a larger number of nodes belonging to the minority class. The mechanism for generating an adjacency matrix, through adaptive sparsity, enhances flexibility in its structure. The feature generation module, called multidimensional features generator (MFG) to generate node features along with topological information. Labels are transformed into embedding vectors, serving as condition
    
[^3]: SoftMAC：基于预测接触模型和与关节刚体和衣物双向耦合的可微软体仿真

    SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes

    [https://arxiv.org/abs/2312.03297](https://arxiv.org/abs/2312.03297)

    SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。

    

    可微物理仿真通过基于梯度的优化，显著提高了解决机器人相关问题的效率。为在各种机器人操纵场景中应用可微仿真，一个关键挑战是将各种材料集成到统一框架中。我们提出了SoftMAC，一个可微仿真框架，将软体与关节刚体和衣物耦合在一起。SoftMAC使用基于连续力学的材料点法来模拟软体。我们提出了一种新颖的基于预测的MPM接触模型，有效减少了穿透，而不会引入其他异常现象，如不自然的反弹。为了将MPM粒子与可变形和非体积衣物网格耦合，我们还提出了一种穿透追踪算法，重建局部区域的有符号距离场。

    arXiv:2312.03297v2 Announce Type: replace-cross  Abstract: Differentiable physics simulation provides an avenue to tackle previously intractable challenges through gradient-based optimization, thereby greatly improving the efficiency of solving robotics-related problems. To apply differentiable simulation in diverse robotic manipulation scenarios, a key challenge is to integrate various materials in a unified framework. We present SoftMAC, a differentiable simulation framework that couples soft bodies with articulated rigid bodies and clothes. SoftMAC simulates soft bodies with the continuum-mechanics-based Material Point Method (MPM). We provide a novel forecast-based contact model for MPM, which effectively reduces penetration without introducing other artifacts like unnatural rebound. To couple MPM particles with deformable and non-volumetric clothes meshes, we also propose a penetration tracing algorithm that reconstructs the signed distance field in local area. Diverging from prev
    
[^4]: 学习视觉连接动作和其效果

    Learning to Visually Connect Actions and their Effects. (arXiv:2401.10805v1 [cs.CV])

    [http://arxiv.org/abs/2401.10805](http://arxiv.org/abs/2401.10805)

    该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。

    

    在这项工作中，我们引入了视觉连接动作和其效果（CATE）的新概念，用于视频理解。CATE可以在任务规划和从示范中学习等领域中应用。我们提出了不同基于CATE的任务形式，如动作选择和动作指定，其中视频理解模型以语义和细粒度的方式连接动作和效果。我们观察到不同的形式产生了捕捉直观动作特性的表示。我们还设计了各种基线模型用于动作选择和动作指定。尽管任务具有直观性，但我们观察到模型困难重重，人类表现明显优于它们。本研究旨在为未来的努力奠定基础，展示了连接视频理解中动作和效果的灵活性和多功能性，希望能激发出高级形式和模型的灵感。

    In this work, we introduce the novel concept of visually Connecting Actions and Their Effects (CATE) in video understanding. CATE can have applications in areas like task planning and learning from demonstration. We propose different CATE-based task formulations, such as action selection and action specification, where video understanding models connect actions and effects at semantic and fine-grained levels. We observe that different formulations produce representations capturing intuitive action properties. We also design various baseline models for action selection and action specification. Despite the intuitive nature of the task, we observe that models struggle, and humans outperform them by a large margin. The study aims to establish a foundation for future efforts, showcasing the flexibility and versatility of connecting actions and effects in video understanding, with the hope of inspiring advanced formulations and models.
    

