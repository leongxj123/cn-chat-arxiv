# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploiting Symmetry in Dynamics for Model-Based Reinforcement Learning with Asymmetric Rewards](https://arxiv.org/abs/2403.19024) | 本文扩展了强化学习和控制理论中对称技术的应用范围，通过利用动态对称性学习动力学模型，而不要求奖励具有相同的对称性。 |
| [^2] | [PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning](https://arxiv.org/abs/2403.00929) | PRIME是一个基于行为原语设计的框架，通过将任务分解为原语序列并学习高级控制策略，显著提高了多阶段操作任务的性能表现。 |
| [^3] | [DexCatch: Learning to Catch Arbitrary Objects with Dexterous Hands.](http://arxiv.org/abs/2310.08809) | 本论文提出了一种稳定约束强化学习算法（SCRL），用于学习用灵巧的手捕捉多样化的物体。该算法在基线方法上取得了很大的优势，并且在未见过的物体上表现出了强大的零-shot迁移性能。 |
| [^4] | [CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving.](http://arxiv.org/abs/2310.07794) | CRITERIA是一种新的基准测试方法，用于评估自动驾驶轨迹预测模型。它通过精细排名预测，提供了关于模型性能在不同情况下的洞察。 |
| [^5] | [Using Implicit Behavior Cloning and Dynamic Movement Primitive to Facilitate Reinforcement Learning for Robot Motion Planning.](http://arxiv.org/abs/2307.16062) | 本文提出了一种利用隐式行为克隆和动态运动原语来促进机器人运动规划的强化学习方法。通过利用人类示范数据提高训练速度，以及将运动规划转化为更简单的规划空间，该方法在仿真和实际机器人实验中展示了更快的训练速度和更高的得分。 |

# 详细

[^1]: 利用动态对称性进行基于模型的非对称奖励强化学习

    Exploiting Symmetry in Dynamics for Model-Based Reinforcement Learning with Asymmetric Rewards

    [https://arxiv.org/abs/2403.19024](https://arxiv.org/abs/2403.19024)

    本文扩展了强化学习和控制理论中对称技术的应用范围，通过利用动态对称性学习动力学模型，而不要求奖励具有相同的对称性。

    

    强化学习中最近的工作利用模型中的对称性来提高策略训练的采样效率。一个常用的简化假设是动力学和奖励都表现出相同的对称性。然而，在许多真实环境中，动力学模型表现出与奖励模型独立的对称性：奖励可能不满足与动力学相同的对称性。本文探讨了只假定动力学表现出对称性的情况，扩展了强化学习和控制理论学习中可应用对称技术的问题范围。我们利用卡塔恩移动框架方法引入一种学习动力学的技术，通过构造，这种动力学表现出指定的对称性。我们通过数值实验展示了所提出的方法学到了更准确的动态模型。

    arXiv:2403.19024v1 Announce Type: cross  Abstract: Recent work in reinforcement learning has leveraged symmetries in the model to improve sample efficiency in training a policy. A commonly used simplifying assumption is that the dynamics and reward both exhibit the same symmetry. However, in many real-world environments, the dynamical model exhibits symmetry independent of the reward model: the reward may not satisfy the same symmetries as the dynamics. In this paper, we investigate scenarios where only the dynamics are assumed to exhibit symmetry, extending the scope of problems in reinforcement learning and learning in control theory where symmetry techniques can be applied. We use Cartan's moving frame method to introduce a technique for learning dynamics which, by construction, exhibit specified symmetries. We demonstrate through numerical experiments that the proposed method learns a more accurate dynamical model.
    
[^2]: 利用行为原语搭建任务的框架以提高数据效率的模仿学习

    PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-Efficient Imitation Learning

    [https://arxiv.org/abs/2403.00929](https://arxiv.org/abs/2403.00929)

    PRIME是一个基于行为原语设计的框架，通过将任务分解为原语序列并学习高级控制策略，显著提高了多阶段操作任务的性能表现。

    

    模仿学习已经显示出巨大潜力，可以让机器人学会复杂的操作行为。然而，在长期任务中，这些算法受到高样本复杂度的困扰，因为复合误差会在任务时段内累积。我们提出了PRIME（基于行为原语的数据效率模仿），这是一个基于行为原语的框架，旨在提高模仿学习的数据效率。PRIME通过将任务演示分解为原语序列来搭建机器人任务，然后通过模仿学习学习一个高级控制策略来对原语序列进行排序。我们的实验证明，PRIME在多阶段操作任务中实现了显著的性能提升，在模拟环境中的成功率比最先进的基线高出10-34％，在实际硬件上高出20-48％。

    arXiv:2403.00929v1 Announce Type: cross  Abstract: Imitation learning has shown great potential for enabling robots to acquire complex manipulation behaviors. However, these algorithms suffer from high sample complexity in long-horizon tasks, where compounding errors accumulate over the task horizons. We present PRIME (PRimitive-based IMitation with data Efficiency), a behavior primitive-based framework designed for improving the data efficiency of imitation learning. PRIME scaffolds robot tasks by decomposing task demonstrations into primitive sequences, followed by learning a high-level control policy to sequence primitives through imitation learning. Our experiments demonstrate that PRIME achieves a significant performance improvement in multi-stage manipulation tasks, with 10-34% higher success rates in simulation over state-of-the-art baselines and 20-48% on physical hardware.
    
[^3]: DexCatch: 学习用灵巧的手捕捉任意物体

    DexCatch: Learning to Catch Arbitrary Objects with Dexterous Hands. (arXiv:2310.08809v1 [cs.RO])

    [http://arxiv.org/abs/2310.08809](http://arxiv.org/abs/2310.08809)

    本论文提出了一种稳定约束强化学习算法（SCRL），用于学习用灵巧的手捕捉多样化的物体。该算法在基线方法上取得了很大的优势，并且在未见过的物体上表现出了强大的零-shot迁移性能。

    

    在机器人领域中，实现类似于人类灵巧操纵的能力仍然是一个关键的研究领域。现有的研究主要集中在提高拿取和放置任务的成功率上。与拿取和放置相比，抛接行为有潜力在无需将物体运送到目的地的情况下提高拿取速度。然而，动态的灵巧操纵由于大量的动态接触而面临着稳定控制的重大挑战。在本文中，我们提出了一种稳定约束强化学习（SCRL）算法，用于学习用灵巧的手捕捉多样化的物体。该算法在基线方法上取得了很大的优势，并且学习到的策略在未见过的物体上表现出了强大的零-shot迁移性能。值得注意的是，即使手中的物体面向侧面非常不稳定，由于缺乏来自手掌的支撑，我们的方法仍然可以在最具挑战的任务中取得很高的成功率。我们还展示了学到的行为的视频演示和合作结果。

    Achieving human-like dexterous manipulation remains a crucial area of research in robotics. Current research focuses on improving the success rate of pick-and-place tasks. Compared with pick-and-place, throw-catching behavior has the potential to increase picking speed without transporting objects to their destination. However, dynamic dexterous manipulation poses a major challenge for stable control due to a large number of dynamic contacts. In this paper, we propose a Stability-Constrained Reinforcement Learning (SCRL) algorithm to learn to catch diverse objects with dexterous hands. The SCRL algorithm outperforms baselines by a large margin, and the learned policies show strong zero-shot transfer performance on unseen objects. Remarkably, even though the object in a hand facing sideward is extremely unstable due to the lack of support from the palm, our method can still achieve a high level of success in the most challenging task. Video demonstrations of learned behaviors and the co
    
[^4]: CRITERIA：一种评估自动驾驶轨迹预测模型的新基准方法

    CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving. (arXiv:2310.07794v1 [cs.CV])

    [http://arxiv.org/abs/2310.07794](http://arxiv.org/abs/2310.07794)

    CRITERIA是一种新的基准测试方法，用于评估自动驾驶轨迹预测模型。它通过精细排名预测，提供了关于模型性能在不同情况下的洞察。

    

    基准测试是评估自动驾驶轨迹预测模型常用的方法。现有的基准测试依赖于数据集，这些数据集对于较常见的情况（如巡航）存在偏差，并通过对所有情况进行平均计算的基于距离的度量。这种方法很少能提供有关模型性能的洞察，无论是在不同情况下它们能否良好处理，还是它们的输出是否允许和多样化。虽然存在一些用于衡量轨迹可允许性和多样性的补充指标，但它们受到偏见的影响，如轨迹长度。在本文中，我们提出了一种新的基准测试方法（CRITERIA），用于评估轨迹预测方法。特别地，我们提出了一种根据道路结构、模型性能和数据特性提取驾驶场景的方法，以进行精细排名预测。

    Benchmarking is a common method for evaluating trajectory prediction models for autonomous driving. Existing benchmarks rely on datasets, which are biased towards more common scenarios, such as cruising, and distance-based metrics that are computed by averaging over all scenarios. Following such a regiment provides a little insight into the properties of the models both in terms of how well they can handle different scenarios and how admissible and diverse their outputs are. There exist a number of complementary metrics designed to measure the admissibility and diversity of trajectories, however, they suffer from biases, such as length of trajectories.  In this paper, we propose a new benChmarking paRadIgm for evaluaTing trajEctoRy predIction Approaches (CRITERIA). Particularly, we propose 1) a method for extracting driving scenarios at varying levels of specificity according to the structure of the roads, models' performance, and data properties for fine-grained ranking of prediction 
    
[^5]: 使用隐式行为克隆和动态运动原语，促进机器人运动规划的强化学习

    Using Implicit Behavior Cloning and Dynamic Movement Primitive to Facilitate Reinforcement Learning for Robot Motion Planning. (arXiv:2307.16062v1 [cs.RO])

    [http://arxiv.org/abs/2307.16062](http://arxiv.org/abs/2307.16062)

    本文提出了一种利用隐式行为克隆和动态运动原语来促进机器人运动规划的强化学习方法。通过利用人类示范数据提高训练速度，以及将运动规划转化为更简单的规划空间，该方法在仿真和实际机器人实验中展示了更快的训练速度和更高的得分。

    

    多自由度机器人的动作规划中，强化学习仍然面临训练速度慢和泛化能力差的问题。本文提出了一种新的基于强化学习的机器人运动规划框架，利用隐式行为克隆和动态运动原语来提高离线策略强化学习代理的训练速度和泛化能力。隐式行为克隆利用人类示范数据提高强化学习的训练速度，而动态运动原语作为一种启发式模型，将运动规划转化为更简单的规划空间。为了支持这一框架，我们还使用拾取-放置实验创建了人类示范数据集，供类似研究使用。在仿真比较实验中，我们发现该方法相比传统的强化学习代理具有更快的训练速度和更高的得分。在实际机器人实验中，该方法展示了在简单组装任务中的适用性。本文提供了一种新的方法，以提高机器人运动规划强化学习的训练速度和泛化能力。

    Reinforcement learning (RL) for motion planning of multi-degree-of-freedom robots still suffers from low efficiency in terms of slow training speed and poor generalizability. In this paper, we propose a novel RL-based robot motion planning framework that uses implicit behavior cloning (IBC) and dynamic movement primitive (DMP) to improve the training speed and generalizability of an off-policy RL agent. IBC utilizes human demonstration data to leverage the training speed of RL, and DMP serves as a heuristic model that transfers motion planning into a simpler planning space. To support this, we also create a human demonstration dataset using a pick-and-place experiment that can be used for similar studies. Comparison studies in simulation reveal the advantage of the proposed method over the conventional RL agents with faster training speed and higher scores. A real-robot experiment indicates the applicability of the proposed method to a simple assembly task. Our work provides a novel pe
    

