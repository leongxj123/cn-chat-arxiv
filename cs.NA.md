# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generating synthetic data for neural operators.](http://arxiv.org/abs/2401.02398) | 该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。 |

# 详细

[^1]: 生成神经算子的合成数据

    Generating synthetic data for neural operators. (arXiv:2401.02398v1 [cs.LG])

    [http://arxiv.org/abs/2401.02398](http://arxiv.org/abs/2401.02398)

    该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。

    

    近期文献中的许多发展展示了深度学习在获取偏微分方程（PDEs）的数值解方面的潜力，这超出了当前数值求解器的能力。然而，数据驱动的神经算子都存在同样的问题：训练网络所需的数据依赖于传统的数值求解器，如有限差分或有限元等。本文提出了一种新方法，用于生成合成的函数训练数据，而无需数值求解PDE。我们的方法很简单：我们从已知解位于的经典理论解空间（例如$H_0^1(\Omega)$）中抽取大量独立同分布的“随机函数”$u_j$，然后将每个随机解方案代入方程并获得相应的右侧函数$f_j$，将$(f_j, u_j)_{j=1}^N$作为监督训练数据。

    Numerous developments in the recent literature show the promising potential of deep learning in obtaining numerical solutions to partial differential equations (PDEs) beyond the reach of current numerical solvers. However, data-driven neural operators all suffer from the same problem: the data needed to train a network depends on classical numerical solvers such as finite difference or finite element, among others. In this paper, we propose a new approach to generating synthetic functional training data that does not require solving a PDE numerically. The way we do this is simple: we draw a large number $N$ of independent and identically distributed `random functions' $u_j$ from the underlying solution space (e.g., $H_0^1(\Omega)$) in which we know the solution lies according to classical theory. We then plug each such random candidate solution into the equation and get a corresponding right-hand side function $f_j$ for the equation, and consider $(f_j, u_j)_{j=1}^N$ as supervised trai
    

