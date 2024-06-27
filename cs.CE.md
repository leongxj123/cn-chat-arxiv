# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning physics-based reduced models from data for the Hasegawa-Wakatani equations.](http://arxiv.org/abs/2401.05972) | 本文提出使用非侵入式科学机器学习（SciML）中的operator inference（OpInf）方法，从数据中构建基于物理的低成本简化模型（ROMs），用于非线性等离子体湍流模拟的Hasegawa-Wakatani方程。实验证明该方法在处理复杂、非线性和自驱动动力学模型时具有潜力。 |

# 详细

[^1]: 从数据中学习基于物理的简化模型，用于Hasegawa-Wakatani方程

    Learning physics-based reduced models from data for the Hasegawa-Wakatani equations. (arXiv:2401.05972v1 [physics.comp-ph])

    [http://arxiv.org/abs/2401.05972](http://arxiv.org/abs/2401.05972)

    本文提出使用非侵入式科学机器学习（SciML）中的operator inference（OpInf）方法，从数据中构建基于物理的低成本简化模型（ROMs），用于非线性等离子体湍流模拟的Hasegawa-Wakatani方程。实验证明该方法在处理复杂、非线性和自驱动动力学模型时具有潜力。

    

    本文关注非侵入式科学机器学习（SciML）的约减模型（ROMs）构建，用于非线性、混沌等离子体湍流模拟。我们提出使用operator inference（OpInf）从数据中构建基于物理的低成本ROMs用于这种模拟。以Hasegawa-Wakatani（HW）方程为代表，在形成复杂、非线性和自驱动动力学的情况下考察了OpInf构建准确ROMs的潜力，并进行了两组实验。第一组实验利用通过直接数值模拟从特定初值条件开始的HW方程获得的数据，进行OpInf ROMs的训练以实现超越训练时间范围的预测。在更具挑战性的第二组实验中，我们使用相同的数据集对ROMs进行训练。

    This paper focuses on the construction of non-intrusive Scientific Machine Learning (SciML) Reduced-Order Models (ROMs) for nonlinear, chaotic plasma turbulence simulations. In particular, we propose using Operator Inference (OpInf) to build low-cost physics-based ROMs from data for such simulations. As a representative example, we focus on the Hasegawa-Wakatani (HW) equations used for modeling two-dimensional electrostatic drift-wave plasma turbulence. For a comprehensive perspective of the potential of OpInf to construct accurate ROMs for this model, we consider a setup for the HW equations that leads to the formation of complex, nonlinear, and self-driven dynamics, and perform two sets of experiments. We first use the data obtained via a direct numerical simulation of the HW equations starting from a specific initial condition and train OpInf ROMs for predictions beyond the training time horizon. In the second, more challenging set of experiments, we train ROMs using the same datase
    

