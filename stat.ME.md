# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel.](http://arxiv.org/abs/2311.01762) | 本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。 |

# 详细

[^1]: 使用梯度下降法解决非常数核的核岭回归

    Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel. (arXiv:2311.01762v1 [stat.ML])

    [http://arxiv.org/abs/2311.01762](http://arxiv.org/abs/2311.01762)

    本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。

    

    核岭回归（KRR）是线性岭回归的推广，它在数据中是非线性的，但在参数中是线性的。解决方案可以通过闭式解获得，其中包括矩阵求逆，也可以通过梯度下降迭代获得。本文研究了在训练过程中改变核函数的方法。我们从理论上探讨了这对模型复杂性和泛化性能的影响。基于我们的发现，我们提出了一种用于平移不变核的带宽更新方案，其中带宽在训练过程中逐渐减小至零，从而避免了超参数选择的需要。我们在真实和合成数据上展示了在训练过程中逐渐减小带宽的优于使用常数带宽，通过交叉验证和边缘似然最大化选择的带宽。我们还从理论和实证上证明了使用逐渐减小的带宽时，我们能够...

    Kernel ridge regression, KRR, is a generalization of linear ridge regression that is non-linear in the data, but linear in the parameters. The solution can be obtained either as a closed-form solution, which includes a matrix inversion, or iteratively through gradient descent. Using the iterative approach opens up for changing the kernel during training, something that is investigated in this paper. We theoretically address the effects this has on model complexity and generalization. Based on our findings, we propose an update scheme for the bandwidth of translational-invariant kernels, where we let the bandwidth decrease to zero during training, thus circumventing the need for hyper-parameter selection. We demonstrate on real and synthetic data how decreasing the bandwidth during training outperforms using a constant bandwidth, selected by cross-validation and marginal likelihood maximization. We also show theoretically and empirically that using a decreasing bandwidth, we are able to
    

