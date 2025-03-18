# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer](https://arxiv.org/abs/2402.06783) | 本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。 |
| [^2] | [Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs.](http://arxiv.org/abs/2401.12243) | Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。 |
| [^3] | [SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation.](http://arxiv.org/abs/2401.08957) | 本论文提出了一种新型框架SWBT，能够在机器人操作任务中有效地从专家演示和不完美演示中学习，而无需与环境进行交互。这是第一个将不完美演示整合到离线模仿学习设置中的机器人操作任务的研究。 |

# 详细

[^1]: 学习教学：改善教师-学生学习中的样本效率，实现从模拟到现实的迁移

    Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer

    [https://arxiv.org/abs/2402.06783](https://arxiv.org/abs/2402.06783)

    本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。

    

    模拟到现实（sim-to-real）的迁移是机器人学习中的一个基本问题。域随机化是一种在训练过程中添加随机性的强大技术，可以有效解决模拟与现实之间的差距。然而，观测中的噪声使得学习变得更加困难。最近的研究表明，采用教师-学生学习范式可以加速随机化环境中的训练。通过使用特权信息进行学习，教师智能体可以指导学生智能体在噪声环境中操作。然而，这种方法通常不是样本效率的，因为在训练学生智能体时完全舍弃了教师智能体收集的经验，浪费了环境所透露的信息。在这项工作中，我们通过提出一个名为学习教学（L2T）的样本效率学习框架来扩展教师-学生学习范式，该框架可以回收教师智能体收集的经验。我们观察到，对于一对教师-学生智能体，环境的动态特性对两者都有重要影响。

    Simulation-to-reality (sim-to-real) transfer is a fundamental problem for robot learning. Domain Randomization, which adds randomization during training, is a powerful technique that effectively addresses the sim-to-real gap. However, the noise in observations makes learning significantly harder. Recently, studies have shown that employing a teacher-student learning paradigm can accelerate training in randomized environments. Learned with privileged information, a teacher agent can instruct the student agent to operate in noisy environments. However, this approach is often not sample efficient as the experience collected by the teacher is discarded completely when training the student, wasting information revealed by the environment. In this work, we extend the teacher-student learning paradigm by proposing a sample efficient learning framework termed Learn to Teach (L2T) that recycles experience collected by the teacher agent. We observe that the dynamics of the environments for both 
    
[^2]: Constraint-Generation Policy Optimization (CGPO): 针对混合离散连续MDPs中的策略优化的非线性规划

    Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs. (arXiv:2401.12243v1 [math.OC])

    [http://arxiv.org/abs/2401.12243](http://arxiv.org/abs/2401.12243)

    Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。

    

    我们提出了Constraint-Generation Policy Optimization (CGPO)方法，用于在混合离散连续Markov Decision Processes (DC-MDPs)中优化策略参数。CGPO不仅能够提供有界的策略误差保证，覆盖具有表达能力的非线性动力学的无数初始状态范围的DC-MDPs，而且在结束时可以明确地推导出最优策略。此外，CGPO还能够生成最坏情况的状态轨迹来诊断策略缺陷，并提供最优行动的反事实解释。为了实现这些结果，CGPO提出了一个双层的混合整数非线性优化框架，用于在定义的表达能力类别（即分段(非)线性）内优化策略，并将其转化为一个最优的约束生成方法，通过对抗性生成最坏情况的状态轨迹。此外，借助现代非线性优化器，CGPO可以获得解决方案。

    We propose Constraint-Generation Policy Optimization (CGPO) for optimizing policy parameters within compact and interpretable policy classes for mixed discrete-continuous Markov Decision Processes (DC-MDPs). CGPO is not only able to provide bounded policy error guarantees over an infinite range of initial states for many DC-MDPs with expressive nonlinear dynamics, but it can also provably derive optimal policies in cases where it terminates with zero error. Furthermore, CGPO can generate worst-case state trajectories to diagnose policy deficiencies and provide counterfactual explanations of optimal actions. To achieve such results, CGPO proposes a bi-level mixed-integer nonlinear optimization framework for optimizing policies within defined expressivity classes (i.e. piecewise (non)-linear) and reduces it to an optimal constraint generation methodology that adversarially generates worst-case state trajectories. Furthermore, leveraging modern nonlinear optimizers, CGPO can obtain soluti
    
[^3]: SWBT：具有不完美演示的相似性加权行为转换器用于机器人操作

    SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation. (arXiv:2401.08957v1 [cs.RO])

    [http://arxiv.org/abs/2401.08957](http://arxiv.org/abs/2401.08957)

    本论文提出了一种新型框架SWBT，能够在机器人操作任务中有效地从专家演示和不完美演示中学习，而无需与环境进行交互。这是第一个将不完美演示整合到离线模仿学习设置中的机器人操作任务的研究。

    

    模仿学习旨在从专家演示中学习最佳控制策略，已成为机器人操作任务的有效方法。然而，先前的模仿学习方法要么仅使用昂贵的专家演示并忽略不完美的演示，要么依赖于与环境的交互和从在线经验中学习。在机器人操作的背景下，我们旨在克服上述两个挑战，并提出了一种名为Similarity Weighted Behavior Transformer（SWBT）的新型框架。SWBT能够有效地从专家演示和不完美演示中学习，而无需与环境进行交互。我们揭示了易获取的不完美演示，如正向和反向动力学，通过学习有益信息显著增强了网络。据我们所知，我们是第一个尝试将不完美演示整合到离线模仿学习设置中的机器人操作任务中的研究。在ManiSkill2 bench上进行了大量实验。

    Imitation learning (IL), aiming to learn optimal control policies from expert demonstrations, has been an effective method for robot manipulation tasks. However, previous IL methods either only use expensive expert demonstrations and omit imperfect demonstrations or rely on interacting with the environment and learning from online experiences. In the context of robotic manipulation, we aim to conquer the above two challenges and propose a novel framework named Similarity Weighted Behavior Transformer (SWBT). SWBT effectively learn from both expert and imperfect demonstrations without interaction with environments. We reveal that the easy-to-get imperfect demonstrations, such as forward and inverse dynamics, significantly enhance the network by learning fruitful information. To the best of our knowledge, we are the first to attempt to integrate imperfect demonstrations into the offline imitation learning setting for robot manipulation tasks. Extensive experiments on the ManiSkill2 bench
    

