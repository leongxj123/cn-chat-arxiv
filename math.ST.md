# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces](https://arxiv.org/abs/2402.04691) | 本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。 |
| [^2] | [Functional Limit Theorems for Hawkes Processes.](http://arxiv.org/abs/2401.11495) | Hawkes过程的长期行为由子事件的平均数量和离散程度决定。对于亚临界过程，提供了FLLNs和FCLTs，具体形式取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，不存在功能中心极限定理。通过缩放后的强度过程和缩放后的Hawkes过程的分布与极限过程之间的Wasserstein距离的上界，给出了收敛速度。具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性，功能极限定理成立。 |

# 详细

[^1]: 在一般希尔伯特空间中使用随机梯度下降学习算子

    Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces

    [https://arxiv.org/abs/2402.04691](https://arxiv.org/abs/2402.04691)

    本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。

    

    本研究探讨了利用随机梯度下降（SGD）在一般希尔伯特空间中学习算子的方法。我们提出了针对目标算子的弱和强规则条件，以描述其内在结构和复杂性。在这些条件下，我们建立了SGD算法的收敛速度的上界，并进行了极小值下界分析，进一步说明我们的收敛分析和规则条件定量地刻画了使用SGD算法解决算子学习问题的可行性。值得强调的是，我们的收敛分析对于非线性算子学习仍然有效。我们证明了SGD估计器将收敛于非线性目标算子的最佳线性近似。此外，将我们的分析应用于基于矢量值和实值再生核希尔伯特空间的算子学习问题，产生了新的收敛结果，从而完善了现有文献的结论。

    This study investigates leveraging stochastic gradient descent (SGD) to learn operators between general Hilbert spaces. We propose weak and strong regularity conditions for the target operator to depict its intrinsic structure and complexity. Under these conditions, we establish upper bounds for convergence rates of the SGD algorithm and conduct a minimax lower bound analysis, further illustrating that our convergence analysis and regularity conditions quantitatively characterize the tractability of solving operator learning problems using the SGD algorithm. It is crucial to highlight that our convergence analysis is still valid for nonlinear operator learning. We show that the SGD estimator will converge to the best linear approximation of the nonlinear target operator. Moreover, applying our analysis to operator learning problems based on vector-valued and real-valued reproducing kernel Hilbert spaces yields new convergence results, thereby refining the conclusions of existing litera
    
[^2]: Hawkes过程的功能极限定理

    Functional Limit Theorems for Hawkes Processes. (arXiv:2401.11495v1 [math.PR])

    [http://arxiv.org/abs/2401.11495](http://arxiv.org/abs/2401.11495)

    Hawkes过程的长期行为由子事件的平均数量和离散程度决定。对于亚临界过程，提供了FLLNs和FCLTs，具体形式取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，不存在功能中心极限定理。通过缩放后的强度过程和缩放后的Hawkes过程的分布与极限过程之间的Wasserstein距离的上界，给出了收敛速度。具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性，功能极限定理成立。

    

    我们证明Hawkes过程的长期行为完全由子事件的平均数量和离散程度确定。对于亚临界过程，我们在过程的核函数的最小条件下提供了FLLNs和FCLTs，极限定理的具体形式严重取决于子事件的离散程度。对于具有弱离散子事件的临界Hawkes过程，功能中心极限定理不成立。相反，我们证明了经过缩放的强度过程和经过缩放的Hawkes过程分别行为类似于无均值回归的CIR过程和整合的CIR过程。通过建立缩放的Hawkes过程的分布与相应极限过程之间的Wasserstein距离的上界，我们给出了收敛速度。相反，具有高度离散子事件的临界Hawkes过程与亚临界过程共享许多属性。特别是，功能极限定理成立。然而，与亚临界过程不同的是，亚临界过程没有离散子事件，具有强离散子事件的临界Hawkes过程没有功能中心极限定理。

    We prove that the long-run behavior of Hawkes processes is fully determined by the average number and the dispersion of child events. For subcritical processes we provide FLLNs and FCLTs under minimal conditions on the kernel of the process with the precise form of the limit theorems depending strongly on the dispersion of child events. For a critical Hawkes process with weakly dispersed child events, functional central limit theorems do not hold. Instead, we prove that the rescaled intensity processes and rescaled Hawkes processes behave like CIR-processes without mean-reversion, respectively integrated CIR-processes. We provide the rate of convergence by establishing an upper bound on the Wasserstein distance between the distributions of rescaled Hawkes process and the corresponding limit process. By contrast, critical Hawkes process with heavily dispersed child events share many properties of subcritical ones. In particular, functional limit theorems hold. However, unlike subcritica
    

