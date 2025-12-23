# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs](https://arxiv.org/abs/2403.15676) | 该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。 |
| [^2] | [IT Intrusion Detection Using Statistical Learning and Testbed Measurements](https://arxiv.org/abs/2402.13081) | 该研究通过统计学习方法和基础设施连续测量数据，以及在内部测试台上进行攻击模拟，实现了IT基础设施中的自动入侵检测。 |

# 详细

[^1]: AC4：用于ZKP中电路约束的代数计算检查器

    AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs

    [https://arxiv.org/abs/2403.15676](https://arxiv.org/abs/2403.15676)

    该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。

    

    ZKP系统已经引起了人们的关注，在当代密码学中发挥着基础性作用。 Zk-SNARK协议主导了ZKP的使用，通常通过算术电路编程范式实现。然而，欠约束或过约束的电路可能导致错误。 欠约束的电路指的是缺乏必要约束的电路，导致电路中出现意外解决方案，并导致验证者接受错误见证。 过约束的电路是指约束过度的电路，导致电路缺乏必要的解决方案，并导致验证者接受没有见证，使电路毫无意义。 本文介绍了一种新方法，用于找出ZKP电路中两种不同类型的错误。 该方法涉及将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统。

    arXiv:2403.15676v1 Announce Type: cross  Abstract: ZKP systems have surged attention and held a fundamental role in contemporary cryptography. Zk-SNARK protocols dominate the ZKP usage, often implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. Underconstrained circuits refer to circuits that lack the necessary constraints, resulting in unexpected solutions in the circuit and causing the verifier to accept a bogus witness. Overconstrained circuits refer to circuits that are constrained excessively, resulting in the circuit lacking necessary solutions and causing the verifier to accept no witness, rendering the circuit meaningless. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving polynomial equation systems over a finite field by algebraic computation. T
    
[^2]: 使用统计学习和测试台测量进行IT入侵检测

    IT Intrusion Detection Using Statistical Learning and Testbed Measurements

    [https://arxiv.org/abs/2402.13081](https://arxiv.org/abs/2402.13081)

    该研究通过统计学习方法和基础设施连续测量数据，以及在内部测试台上进行攻击模拟，实现了IT基础设施中的自动入侵检测。

    

    我们研究了IT基础设施中的自动入侵检测，特别是识别攻击开始、攻击类型以及攻击者采取的动作顺序的问题，基于基础设施的连续测量。我们应用统计学习方法，包括隐马尔可夫模型（HMM）、长短期记忆（LSTM）和随机森林分类器（RFC），将观测序列映射到预测攻击动作序列。与大多数相关研究不同，我们拥有丰富的数据来训练模型并评估其预测能力。数据来自我们在内部测试台上生成的跟踪数据，在这里我们对模拟的IT基础设施进行攻击。我们工作的核心是一个机器学习管道，将来自高维观测空间的测量映射到低维空间或少量观测符号的空间。我们研究离线和在线入侵检测

    arXiv:2402.13081v1 Announce Type: new  Abstract: We study automated intrusion detection in an IT infrastructure, specifically the problem of identifying the start of an attack, the type of attack, and the sequence of actions an attacker takes, based on continuous measurements from the infrastructure. We apply statistical learning methods, including Hidden Markov Model (HMM), Long Short-Term Memory (LSTM), and Random Forest Classifier (RFC) to map sequences of observations to sequences of predicted attack actions. In contrast to most related research, we have abundant data to train the models and evaluate their predictive power. The data comes from traces we generate on an in-house testbed where we run attacks against an emulated IT infrastructure. Central to our work is a machine-learning pipeline that maps measurements from a high-dimensional observation space to a space of low dimensionality or to a small set of observation symbols. Investigating intrusions in offline as well as onli
    

