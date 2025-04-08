# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Purpose for Open-Ended Learning Robots: A Computational Taxonomy, Definition, and Operationalisation](https://arxiv.org/abs/2403.02514) | 提出了设定机器人目的的概念，以帮助机器人更加关注获取与目的相关的知识。 |
| [^2] | [SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes](https://arxiv.org/abs/2312.03297) | SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。 |
| [^3] | [Online POMDP Planning with Anytime Deterministic Guarantees.](http://arxiv.org/abs/2310.01791) | 本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。 |

# 详细

[^1]: 为开放式学习机器人设定目的：一个计算分类、定义和操作化

    Purpose for Open-Ended Learning Robots: A Computational Taxonomy, Definition, and Operationalisation

    [https://arxiv.org/abs/2403.02514](https://arxiv.org/abs/2403.02514)

    提出了设定机器人目的的概念，以帮助机器人更加关注获取与目的相关的知识。

    

    arXiv:2403.02514v1 公告类型: 跨领域 摘要: 自主开放式学习(OEL)机器人能够通过与环境的直接交互累积获取新技能和知识，例如依靠内在动机和自动生成的目标的指导。OEL机器人对应用具有很高的相关性，因为它们可以使用自主获取的知识来完成对人类用户有关的任务。然而，OEL机器人面临一个重要限制：这可能导致获取的知识对完成用户任务并不那么重要。本文分析了这个问题的一个可能解决方案，它围绕“目的”这一新概念展开。目的表示设计者和/或用户希望机器人从中获得什么。机器人应使用目的的内部表征，这里称为“愿望”，来将其开放式探索集中于获取与其完成目的相关的知识。这项工作有助于发展一个共同

    arXiv:2403.02514v1 Announce Type: cross  Abstract: Autonomous open-ended learning (OEL) robots are able to cumulatively acquire new skills and knowledge through direct interaction with the environment, for example relying on the guidance of intrinsic motivations and self-generated goals. OEL robots have a high relevance for applications as they can use the autonomously acquired knowledge to accomplish tasks relevant for their human users. OEL robots, however, encounter an important limitation: this may lead to the acquisition of knowledge that is not so much relevant to accomplish the users' tasks. This work analyses a possible solution to this problem that pivots on the novel concept of `purpose'. Purposes indicate what the designers and/or users want from the robot. The robot should use internal representations of purposes, called here `desires', to focus its open-ended exploration towards the acquisition of knowledge relevant to accomplish them. This work contributes to develop a co
    
[^2]: SoftMAC：基于预测接触模型和与关节刚体和衣物双向耦合的可微软体仿真

    SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes

    [https://arxiv.org/abs/2312.03297](https://arxiv.org/abs/2312.03297)

    SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。

    

    可微物理仿真通过基于梯度的优化，显著提高了解决机器人相关问题的效率。为在各种机器人操纵场景中应用可微仿真，一个关键挑战是将各种材料集成到统一框架中。我们提出了SoftMAC，一个可微仿真框架，将软体与关节刚体和衣物耦合在一起。SoftMAC使用基于连续力学的材料点法来模拟软体。我们提出了一种新颖的基于预测的MPM接触模型，有效减少了穿透，而不会引入其他异常现象，如不自然的反弹。为了将MPM粒子与可变形和非体积衣物网格耦合，我们还提出了一种穿透追踪算法，重建局部区域的有符号距离场。

    arXiv:2312.03297v2 Announce Type: replace-cross  Abstract: Differentiable physics simulation provides an avenue to tackle previously intractable challenges through gradient-based optimization, thereby greatly improving the efficiency of solving robotics-related problems. To apply differentiable simulation in diverse robotic manipulation scenarios, a key challenge is to integrate various materials in a unified framework. We present SoftMAC, a differentiable simulation framework that couples soft bodies with articulated rigid bodies and clothes. SoftMAC simulates soft bodies with the continuum-mechanics-based Material Point Method (MPM). We provide a novel forecast-based contact model for MPM, which effectively reduces penetration without introducing other artifacts like unnatural rebound. To couple MPM particles with deformable and non-volumetric clothes meshes, we also propose a penetration tracing algorithm that reconstructs the signed distance field in local area. Diverging from prev
    
[^3]: 具有任意确定性保证的在线POMDP规划

    Online POMDP Planning with Anytime Deterministic Guarantees. (arXiv:2310.01791v1 [cs.AI])

    [http://arxiv.org/abs/2310.01791](http://arxiv.org/abs/2310.01791)

    本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。

    

    在现实场景中，自主智能体经常遇到不确定性并基于不完整信息做出决策。在不确定性下的规划可以使用部分可观察的马尔科夫决策过程（POMDP）进行数学建模。然而，寻找POMDP的最优规划在计算上是昂贵的，只有在小规模任务中可行。近年来，近似算法（如树搜索和基于采样的方法）已经成为解决较大问题的先进POMDP求解器。尽管这些算法有效，但它们仅提供概率性和通常呈现渐进性保证，这是由于它们依赖于采样的缘故。为了解决这些限制，我们推导出一个简化解决方案与理论上最优解之间的确定性关系。首先，我们推导出选择一组观测以在计算每个后验节点时分支的边界。

    Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that is easier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior nod
    

