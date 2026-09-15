# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximation of RKHS Functionals by Neural Networks](https://arxiv.org/abs/2403.12187) | 本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型，并建立了逼近的普适性，推导了逆多重二次、高斯和Sobolev核引起的误差界限，证明神经网络可以准确逼近广义函数线性模型中的回归映射。 |
| [^2] | [Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces](https://arxiv.org/abs/2402.04691) | 本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。 |
| [^3] | [Unfair Utilities and First Steps Towards Improving Them.](http://arxiv.org/abs/2306.00636) | 该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。 |

# 详细

[^1]: 神经网络逼近RKHS函数型

    Approximation of RKHS Functionals by Neural Networks

    [https://arxiv.org/abs/2403.12187](https://arxiv.org/abs/2403.12187)

    本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型，并建立了逼近的普适性，推导了逆多重二次、高斯和Sobolev核引起的误差界限，证明神经网络可以准确逼近广义函数线性模型中的回归映射。

    

    受到时间序列和图像等丰富功能性数据的启发，人们越来越感兴趣将这些数据整合到神经网络中，并从函数空间到R（即函数型）学习映射。本文研究了使用神经网络逼近再生核希尔伯特空间（RKHS）上的函数型。我们建立了对RKHS上函数型逼近的普适性。具体来说，我们推导了通过逆多重二次、高斯和Sobolev核引起的明确误差界限。此外，我们将我们的研究应用于函数回归，证明了神经网络可以准确逼近广义函数线性模型中的回归映射。现有的功能性学习作品需要积分型基函数展开与一组预定义的基函数。通过在RKHS中利用插值正交投影，我们提出的网络是...

    arXiv:2403.12187v1 Announce Type: cross  Abstract: Motivated by the abundance of functional data such as time series and images, there has been a growing interest in integrating such data into neural networks and learning maps from function spaces to R (i.e., functionals). In this paper, we study the approximation of functionals on reproducing kernel Hilbert spaces (RKHS's) using neural networks. We establish the universality of the approximation of functionals on the RKHS's. Specifically, we derive explicit error bounds for those induced by inverse multiquadric, Gaussian, and Sobolev kernels. Moreover, we apply our findings to functional regression, proving that neural networks can accurately approximate the regression maps in generalized functional linear models. Existing works on functional learning require integration-type basis function expansions with a set of pre-specified basis functions. By leveraging the interpolating orthogonal projections in RKHS's, our proposed network is 
    
[^2]: 在一般希尔伯特空间中使用随机梯度下降学习算子

    Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces

    [https://arxiv.org/abs/2402.04691](https://arxiv.org/abs/2402.04691)

    本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。

    

    本研究探讨了利用随机梯度下降（SGD）在一般希尔伯特空间中学习算子的方法。我们提出了针对目标算子的弱和强规则条件，以描述其内在结构和复杂性。在这些条件下，我们建立了SGD算法的收敛速度的上界，并进行了极小值下界分析，进一步说明我们的收敛分析和规则条件定量地刻画了使用SGD算法解决算子学习问题的可行性。值得强调的是，我们的收敛分析对于非线性算子学习仍然有效。我们证明了SGD估计器将收敛于非线性目标算子的最佳线性近似。此外，将我们的分析应用于基于矢量值和实值再生核希尔伯特空间的算子学习问题，产生了新的收敛结果，从而完善了现有文献的结论。

    This study investigates leveraging stochastic gradient descent (SGD) to learn operators between general Hilbert spaces. We propose weak and strong regularity conditions for the target operator to depict its intrinsic structure and complexity. Under these conditions, we establish upper bounds for convergence rates of the SGD algorithm and conduct a minimax lower bound analysis, further illustrating that our convergence analysis and regularity conditions quantitatively characterize the tractability of solving operator learning problems using the SGD algorithm. It is crucial to highlight that our convergence analysis is still valid for nonlinear operator learning. We show that the SGD estimator will converge to the best linear approximation of the nonlinear target operator. Moreover, applying our analysis to operator learning problems based on vector-valued and real-valued reproducing kernel Hilbert spaces yields new convergence results, thereby refining the conclusions of existing litera
    
[^3]: 不公平的实用程序及其改进的第一步

    Unfair Utilities and First Steps Towards Improving Them. (arXiv:2306.00636v1 [stat.ML])

    [http://arxiv.org/abs/2306.00636](http://arxiv.org/abs/2306.00636)

    该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。

    

    许多公平标准对政策或预测器的选择进行限制。在这项工作中，我们提出了一个不同的思考公平的框架：我们考虑政策正在优化哪个效用，而不是限制政策或预测器的选择。我们定义了信息价值公平，并建议不使用不满足此标准的实用程序。我们描述了如何修改实用程序以满足这种公平标准，并讨论了这可能对相应的最优政策产生的影响。

    Many fairness criteria constrain the policy or choice of predictors. In this work, we propose a different framework for thinking about fairness: Instead of constraining the policy or choice of predictors, we consider which utility a policy is optimizing for. We define value of information fairness and propose to not use utilities that do not satisfy this criterion. We describe how to modify a utility to satisfy this fairness criterion and discuss the consequences this might have on the corresponding optimal policies.
    

