# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Finite Expression Method for Solving High-Dimensional Committor Problems.](http://arxiv.org/abs/2306.12268) | 本文提出了一种用于解决高维Committor问题的有限表达式方法(FEX)，该方法通过深度神经网络学习最优非线性函数和系数值，能够显著提高计算效果。 |
| [^2] | [Finite Expression Methods for Discovering Physical Laws from Data.](http://arxiv.org/abs/2305.08342) | 本文介绍了一种名为"有限表达法" (FEX) 的深度符号学习方法，通过学习动态数据中PDE解的导数，发现控制方程的解析表达式。相对于其他现有方法，FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。 |

# 详细

[^1]: 一种用于解决高维Committor问题的有限表达式方法

    A Finite Expression Method for Solving High-Dimensional Committor Problems. (arXiv:2306.12268v1 [math.NA])

    [http://arxiv.org/abs/2306.12268](http://arxiv.org/abs/2306.12268)

    本文提出了一种用于解决高维Committor问题的有限表达式方法(FEX)，该方法通过深度神经网络学习最优非线性函数和系数值，能够显著提高计算效果。

    

    转移路径理论（TPT）是一种数学框架，用于量化从选定的亚稳态$A$到$B$之间的稀有转移事件。TPT的核心是Committor函数，其描述了从相空间的任何起始点到达亚稳态$B$之前到达$A$的概率。计算出Committor之后，可以立即找到转换通道和转换速率。Committor是具有适当边界条件的反向Kolmogorov方程的解。然而，在高维情况下，由于需要网格化整个环境空间，解决Committor是一项具有挑战性的任务。在这项工作中，我们探索了有限表达式方法（FEX，Liang和Yang（2022））作为计算Committor的工具。FEX通过涉及一定数量的非线性函数和二进制算术运算的固定有限代数表达式来逼近Committor。最佳的非线性函数、二进制运算和数值系数值通过深度神经网络从训练数据中学习到。我们通过解决多个高维Committor问题，其中包括高达400个维度，展示了FEX的有效性，并且表明FEX显著优于传统的数值方法，如有限元方法和有限差分方法。

    Transition path theory (TPT) is a mathematical framework for quantifying rare transition events between a pair of selected metastable states $A$ and $B$. Central to TPT is the committor function, which describes the probability to hit the metastable state $B$ prior to $A$ from any given starting point of the phase space. Once the committor is computed, the transition channels and the transition rate can be readily found. The committor is the solution to the backward Kolmogorov equation with appropriate boundary conditions. However, solving it is a challenging task in high dimensions due to the need to mesh a whole region of the ambient space. In this work, we explore the finite expression method (FEX, Liang and Yang (2022)) as a tool for computing the committor. FEX approximates the committor by an algebraic expression involving a fixed finite number of nonlinear functions and binary arithmetic operations. The optimal nonlinear functions, the binary operations, and the numerical coeffi
    
[^2]: 从数据中发现物理定律的有限表达方法

    Finite Expression Methods for Discovering Physical Laws from Data. (arXiv:2305.08342v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.08342](http://arxiv.org/abs/2305.08342)

    本文介绍了一种名为"有限表达法" (FEX) 的深度符号学习方法，通过学习动态数据中PDE解的导数，发现控制方程的解析表达式。相对于其他现有方法，FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。

    

    非线性动力学是科学和工程领域中普遍存在的现象。然而，从有限数据中推导出描述非线性动力学的解析表达式仍然具有挑战性。本文将介绍一种新颖的深度符号学习方法，称为"有限表达法" (FEX)，通过学习动态数据中的偏微分方程（PDE）解的导数，利用FEX在包含有限集的解析表达式的函数空间中发现控制方程。我们的数值结果表明，相对于其他现有方法（如PDE-Net, SINDy, GP 和 SPL），我们的FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。此外，结果突显了FEX的灵活性和鲁棒性。

    Nonlinear dynamics is a pervasive phenomenon observed in scientific and engineering disciplines. However, the task of deriving analytical expressions to describe nonlinear dynamics from limited data remains challenging. In this paper, we shall present a novel deep symbolic learning method called the "finite expression method" (FEX) to discover governing equations within a function space containing a finite set of analytic expressions, based on observed dynamic data. The key concept is to employ FEX to generate analytical expressions of the governing equations by learning the derivatives of partial differential equation (PDE) solutions through convolutions. Our numerical results demonstrate that our FEX surpasses other existing methods (such as PDE-Net, SINDy, GP, and SPL) in terms of numerical performance across a range of problems, including time-dependent PDE problems and nonlinear dynamical systems with time-varying coefficients. Moreover, the results highlight FEX's flexibility and
    

