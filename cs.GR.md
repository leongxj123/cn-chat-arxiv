# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions.](http://arxiv.org/abs/2310.00177) | 我们引入了一个用于混合边界条件的泊松方程的神经预处理迭代求解器，核心是一个神经网络，能够近似逆离散结构网格拉普拉斯算子，并且在训练集之外的边界条件下仍然有效。 |

# 详细

[^1]: 一个用于混合迪里切特和诺曼边界条件的神经预处理泊松求解器

    A Neural-preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions. (arXiv:2310.00177v1 [math.NA])

    [http://arxiv.org/abs/2310.00177](http://arxiv.org/abs/2310.00177)

    我们引入了一个用于混合边界条件的泊松方程的神经预处理迭代求解器，核心是一个神经网络，能够近似逆离散结构网格拉普拉斯算子，并且在训练集之外的边界条件下仍然有效。

    

    我们引入了一个神经预处理的迭代求解器，用于具有混合边界条件的泊松方程。泊松方程在科学计算中是普遍存在的：它控制着广泛的物理现象，在许多数值算法中作为子问题出现，并且作为更广泛的椭圆PDE类的模型问题。最流行的泊松离散化方法可以产生大型稀疏线性系统。在高分辨率和对性能至关重要的应用中，迭代求解器结合强大的预处理器可以提供优势。我们求解器的核心是一个神经网络，该网络经过训练可以近似离散结构网格拉普拉斯算子的逆算子，适用于任意形状的域和混合边界条件。我们展示了该问题的结构激发了一种新颖的网络架构，即使在训练集之外的边界条件下，该架构也表现出高效的预处理器。我们展示了在具有挑战性的测试案例上的效果。

    We introduce a neural-preconditioned iterative solver for Poisson equations with mixed boundary conditions. The Poisson equation is ubiquitous in scientific computing: it governs a wide array of physical phenomena, arises as a subproblem in many numerical algorithms, and serves as a model problem for the broader class of elliptic PDEs. The most popular Poisson discretizations yield large sparse linear systems. At high resolution, and for performance-critical applications, iterative solvers can be advantageous for these -- but only when paired with powerful preconditioners. The core of our solver is a neural network trained to approximate the inverse of a discrete structured-grid Laplace operator for a domain of arbitrary shape and with mixed boundary conditions. The structure of this problem motivates a novel network architecture that we demonstrate is highly effective as a preconditioner even for boundary conditions outside the training set. We show that on challenging test cases aris
    

