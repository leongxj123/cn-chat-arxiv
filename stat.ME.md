# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems](https://arxiv.org/abs/2402.02190) | 本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。 |

# 详细

[^1]: 在组合优化问题中寻找多样化解决方案的连续张量放松方法

    Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems

    [https://arxiv.org/abs/2402.02190](https://arxiv.org/abs/2402.02190)

    本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。

    

    在组合优化问题中，寻找最佳解是最常见的目标。然而，在实际场景中，单一解决方案可能不适用，因为目标函数和约束条件只是原始现实世界情况的近似值。为了解决这个问题，寻找具有不同特征的多样化解决方案和约束严重性的变化成为自然的方向。这种策略提供了在后处理过程中选择合适解决方案的灵活性。然而，发现这些多样化解决方案比确定单一解决方案更具挑战性。为了克服这一挑战，本研究引入了连续张量松弛退火 (CTRA) 方法，用于基于无监督学习的组合优化求解器。CTRA通过扩展连续松弛方法，将离散决策变量转换为连续张量，同时解决了多个问题。该方法找到了不同特征的多样化解决方案和约束严重性的变化。

    Finding the best solution is the most common objective in combinatorial optimization (CO) problems. However, a single solution may not be suitable in practical scenarios, as the objective functions and constraints are only approximations of original real-world situations. To tackle this, finding (i) "heterogeneous solutions", diverse solutions with distinct characteristics, and (ii) "penalty-diversified solutions", variations in constraint severity, are natural directions. This strategy provides the flexibility to select a suitable solution during post-processing. However, discovering these diverse solutions is more challenging than identifying a single solution. To overcome this challenge, this study introduces Continual Tensor Relaxation Annealing (CTRA) for unsupervised-learning-based CO solvers. CTRA addresses various problems simultaneously by extending the continual relaxation approach, which transforms discrete decision variables into continual tensors. This method finds heterog
    

