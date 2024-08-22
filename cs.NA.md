# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Operator SVD with Neural Networks via Nested Low-Rank Approximation](https://arxiv.org/abs/2402.03655) | 本文提出了一个新的优化框架，使用嵌套的低秩近似方法通过神经网络实现运算符的奇异值分解。该方法通过无约束优化公式隐式高效地保持学习函数的正交性。 |

# 详细

[^1]: 使用神经网络通过嵌套低秩近似实现运算符的奇异值分解

    Operator SVD with Neural Networks via Nested Low-Rank Approximation

    [https://arxiv.org/abs/2402.03655](https://arxiv.org/abs/2402.03655)

    本文提出了一个新的优化框架，使用嵌套的低秩近似方法通过神经网络实现运算符的奇异值分解。该方法通过无约束优化公式隐式高效地保持学习函数的正交性。

    

    在许多机器学习和科学计算问题中，计算给定线性算子的特征值分解（EVD）或找到其主要特征值和特征函数是一项基础任务。对于高维特征值问题，训练神经网络参数化特征函数被认为是传统数值线性代数技术的有希望的替代方法。本文提出了一个新的优化框架，基于截断奇异值分解的低秩近似表征，并伴随着称为嵌套的学习方法，以正确的顺序学习前L个奇异值和奇异函数。所提出的方法通过无约束优化公式隐式高效地促进了学习函数的正交性，这个公式可以很容易地通过现成的基于梯度的优化算法求解。我们展示了所提出的优化框架在使用案例中的有效性。

    Computing eigenvalue decomposition (EVD) of a given linear operator, or finding its leading eigenvalues and eigenfunctions, is a fundamental task in many machine learning and scientific computing problems. For high-dimensional eigenvalue problems, training neural networks to parameterize the eigenfunctions is considered as a promising alternative to the classical numerical linear algebra techniques. This paper proposes a new optimization framework based on the low-rank approximation characterization of a truncated singular value decomposition, accompanied by new techniques called nesting for learning the top-$L$ singular values and singular functions in the correct order. The proposed method promotes the desired orthogonality in the learned functions implicitly and efficiently via an unconstrained optimization formulation, which is easy to solve with off-the-shelf gradient-based optimization algorithms. We demonstrate the effectiveness of the proposed optimization framework for use cas
    

