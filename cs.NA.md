# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Rounding Implicitly Regularizes Tall-and-Thin Matrices](https://arxiv.org/abs/2403.12278) | 随机舍入技术能有效隐式正则化高瘦矩阵，确保舍入后的矩阵具有完整的列秩。 |
| [^2] | [A physics-informed neural network framework for modeling obstacle-related equations.](http://arxiv.org/abs/2304.03552) | 本文拓展了基于物理知识的神经网络(PINN) 来解决求解障碍物相关的偏微分方程问题，这种类型的问题需要解决数值方法的难度较大，但作者通过对多种情况的研究证明了PINN的有效性。 |

# 详细

[^1]: 随机舍入隐式正则化高瘦矩阵

    Stochastic Rounding Implicitly Regularizes Tall-and-Thin Matrices

    [https://arxiv.org/abs/2403.12278](https://arxiv.org/abs/2403.12278)

    随机舍入技术能有效隐式正则化高瘦矩阵，确保舍入后的矩阵具有完整的列秩。

    

    受到随机舍入在机器学习和大规模深度神经网络模型训练中的流行，我们考虑实矩阵$\mathbf{A}$的随机近似舍入，其中行数远远多于列数。我们提供了新颖的理论证据，并通过大量实验评估支持，高概率下，随机舍入矩阵的最小奇异值远离零--无论$\mathbf{A}$接近奇异还是$\mathbf{A}$奇异。换句话说，随机舍入\textit{隐式正则化}高瘦矩阵$\mathbf{A}$，使得舍入后的版本具有完整的列秩。我们的证明利用了随机矩阵理论中的有力结果，以及随机舍入误差不集中在低维列空间的思想。

    arXiv:2403.12278v1 Announce Type: new  Abstract: Motivated by the popularity of stochastic rounding in the context of machine learning and the training of large-scale deep neural network models, we consider stochastic nearness rounding of real matrices $\mathbf{A}$ with many more rows than columns. We provide novel theoretical evidence, supported by extensive experimental evaluation that, with high probability, the smallest singular value of a stochastically rounded matrix is well bounded away from zero -- regardless of how close $\mathbf{A}$ is to being rank deficient and even if $\mathbf{A}$ is rank-deficient. In other words, stochastic rounding \textit{implicitly regularizes} tall and skinny matrices $\mathbf{A}$ so that the rounded version has full column rank. Our proofs leverage powerful results in random matrix theory, and the idea that stochastic rounding errors do not concentrate in low-dimensional column spaces.
    
[^2]: 基于物理知识的神经网络模型求解障碍物相关方程

    A physics-informed neural network framework for modeling obstacle-related equations. (arXiv:2304.03552v1 [cs.LG])

    [http://arxiv.org/abs/2304.03552](http://arxiv.org/abs/2304.03552)

    本文拓展了基于物理知识的神经网络(PINN) 来解决求解障碍物相关的偏微分方程问题，这种类型的问题需要解决数值方法的难度较大，但作者通过对多种情况的研究证明了PINN的有效性。

    

    深度学习在一些应用中取得了很大成功，但将其用于求解偏微分方程(PDE)　的研究则是近年来的热点，尤其在目前的机器学习库（如TensorFlow或PyTorch）的支持下取得了重大进展。基于物理知识的神经网络（PINN）可通过解析稀疏且噪声数据来求解偏微分方程，是一种有吸引力的工具。本文将拓展PINN来求解障碍物相关PDE，这类方程难度较大，需要可以得到准确解的数值方法。作者在正常和不规则的障碍情况下，对线性和非线性PDE的多个场景进行了演示，证明了所提出的PINNs性能的有效性。

    Deep learning has been highly successful in some applications. Nevertheless, its use for solving partial differential equations (PDEs) has only been of recent interest with current state-of-the-art machine learning libraries, e.g., TensorFlow or PyTorch. Physics-informed neural networks (PINNs) are an attractive tool for solving partial differential equations based on sparse and noisy data. Here extend PINNs to solve obstacle-related PDEs which present a great computational challenge because they necessitate numerical methods that can yield an accurate approximation of the solution that lies above a given obstacle. The performance of the proposed PINNs is demonstrated in multiple scenarios for linear and nonlinear PDEs subject to regular and irregular obstacles.
    

