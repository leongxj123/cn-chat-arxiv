# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer](https://arxiv.org/abs/2402.06783) | 本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。 |

# 详细

[^1]: 学习教学：改善教师-学生学习中的样本效率，实现从模拟到现实的迁移

    Learn to Teach: Improve Sample Efficiency in Teacher-student Learning for Sim-to-Real Transfer

    [https://arxiv.org/abs/2402.06783](https://arxiv.org/abs/2402.06783)

    本文提出了一种样本效率学习框架，名为学习教学（L2T），通过回收教师智能体收集的经验，解决了教师-学生学习中的样本效率问题。

    

    模拟到现实（sim-to-real）的迁移是机器人学习中的一个基本问题。域随机化是一种在训练过程中添加随机性的强大技术，可以有效解决模拟与现实之间的差距。然而，观测中的噪声使得学习变得更加困难。最近的研究表明，采用教师-学生学习范式可以加速随机化环境中的训练。通过使用特权信息进行学习，教师智能体可以指导学生智能体在噪声环境中操作。然而，这种方法通常不是样本效率的，因为在训练学生智能体时完全舍弃了教师智能体收集的经验，浪费了环境所透露的信息。在这项工作中，我们通过提出一个名为学习教学（L2T）的样本效率学习框架来扩展教师-学生学习范式，该框架可以回收教师智能体收集的经验。我们观察到，对于一对教师-学生智能体，环境的动态特性对两者都有重要影响。

    Simulation-to-reality (sim-to-real) transfer is a fundamental problem for robot learning. Domain Randomization, which adds randomization during training, is a powerful technique that effectively addresses the sim-to-real gap. However, the noise in observations makes learning significantly harder. Recently, studies have shown that employing a teacher-student learning paradigm can accelerate training in randomized environments. Learned with privileged information, a teacher agent can instruct the student agent to operate in noisy environments. However, this approach is often not sample efficient as the experience collected by the teacher is discarded completely when training the student, wasting information revealed by the environment. In this work, we extend the teacher-student learning paradigm by proposing a sample efficient learning framework termed Learn to Teach (L2T) that recycles experience collected by the teacher agent. We observe that the dynamics of the environments for both 
    

