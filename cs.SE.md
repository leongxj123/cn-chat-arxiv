# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs](https://arxiv.org/abs/2403.15676) | 该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。 |

# 详细

[^1]: AC4：用于ZKP中电路约束的代数计算检查器

    AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs

    [https://arxiv.org/abs/2403.15676](https://arxiv.org/abs/2403.15676)

    该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。

    

    ZKP系统已经引起了人们的关注，在当代密码学中发挥着基础性作用。 Zk-SNARK协议主导了ZKP的使用，通常通过算术电路编程范式实现。然而，欠约束或过约束的电路可能导致错误。 欠约束的电路指的是缺乏必要约束的电路，导致电路中出现意外解决方案，并导致验证者接受错误见证。 过约束的电路是指约束过度的电路，导致电路缺乏必要的解决方案，并导致验证者接受没有见证，使电路毫无意义。 本文介绍了一种新方法，用于找出ZKP电路中两种不同类型的错误。 该方法涉及将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统。

    arXiv:2403.15676v1 Announce Type: cross  Abstract: ZKP systems have surged attention and held a fundamental role in contemporary cryptography. Zk-SNARK protocols dominate the ZKP usage, often implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. Underconstrained circuits refer to circuits that lack the necessary constraints, resulting in unexpected solutions in the circuit and causing the verifier to accept a bogus witness. Overconstrained circuits refer to circuits that are constrained excessively, resulting in the circuit lacking necessary solutions and causing the verifier to accept no witness, rendering the circuit meaningless. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving polynomial equation systems over a finite field by algebraic computation. T
    

