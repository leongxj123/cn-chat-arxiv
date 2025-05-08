# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Double Cross-fit Doubly Robust Estimators: Beyond Series Regression](https://arxiv.org/abs/2403.15175) | 双交叉固定双稳健估计器针对因果推断中的预期条件协方差进行了研究，通过拆分训练数据并在独立样本上下调nuisance函数估计器，结构无关的错误分析以及更强假设的结果，提出了更精确的DCDR估计器。 |
| [^2] | [Functional Partial Least-Squares: Optimal Rates and Adaptation](https://arxiv.org/abs/2402.11134) | 该论文提出了一种新的函数偏最小二乘估计器，其在一类椭球上实现了（近乎）最优的收敛速率，并引入了适应未知逆问题度的提前停止规则。 |

# 详细

[^1]: 双交叉固定双稳健估计器：超越串行回归

    Double Cross-fit Doubly Robust Estimators: Beyond Series Regression

    [https://arxiv.org/abs/2403.15175](https://arxiv.org/abs/2403.15175)

    双交叉固定双稳健估计器针对因果推断中的预期条件协方差进行了研究，通过拆分训练数据并在独立样本上下调nuisance函数估计器，结构无关的错误分析以及更强假设的结果，提出了更精确的DCDR估计器。

    

    具有跨拟合交叉的双稳健估计器因其良好的结构无关错误保证而在因果推断中备受青睐。然而，当存在额外结构，例如H\"{o}lder平滑时，可以通过在独立样本上对训练数据进行拆分和下调nuisance函数估计器来构建更精确的“双交叉固定双稳健”（DCDR）估计器。我们研究了预期条件协方差的DCDR估计器，在因果推断和条件独立性检验中是一个感兴趣的函数，并得出了一系列逐渐更强假设的结果。首先，我们对DCDR估计器提供无需对nuisance函数或它们的估计器做出假设的结构无关错误分析。然后，假设nuisance函数是H\"{o}lder平滑，但不假设知晓真实平滑级别或协变量密度。

    arXiv:2403.15175v1 Announce Type: cross  Abstract: Doubly robust estimators with cross-fitting have gained popularity in causal inference due to their favorable structure-agnostic error guarantees. However, when additional structure, such as H\"{o}lder smoothness, is available then more accurate "double cross-fit doubly robust" (DCDR) estimators can be constructed by splitting the training data and undersmoothing nuisance function estimators on independent samples. We study a DCDR estimator of the Expected Conditional Covariance, a functional of interest in causal inference and conditional independence testing, and derive a series of increasingly powerful results with progressively stronger assumptions. We first provide a structure-agnostic error analysis for the DCDR estimator with no assumptions on the nuisance functions or their estimators. Then, assuming the nuisance functions are H\"{o}lder smooth, but without assuming knowledge of the true smoothness level or the covariate densit
    
[^2]: 函数偏最小二乘法：最优收敛率和自适应性

    Functional Partial Least-Squares: Optimal Rates and Adaptation

    [https://arxiv.org/abs/2402.11134](https://arxiv.org/abs/2402.11134)

    该论文提出了一种新的函数偏最小二乘估计器，其在一类椭球上实现了（近乎）最优的收敛速率，并引入了适应未知逆问题度的提前停止规则。

    

    我们考虑具有标量响应和 Hilbert 空间值预测变量的函数线性回归模型，这是一个众所周知的反问题。我们提出了一个与共轭梯度方法相关的函数偏最小二乘（PLS）估计的新公式。我们将展示该估计器在一类椭球上实现了（近乎）最优的收敛速率，并引入了一个能够适应未知逆问题度的提前停止规则。我们提供了估计器与主成分回归估计器之间的一些理论和仿真比较。

    arXiv:2402.11134v1 Announce Type: cross  Abstract: We consider the functional linear regression model with a scalar response and a Hilbert space-valued predictor, a well-known ill-posed inverse problem. We propose a new formulation of the functional partial least-squares (PLS) estimator related to the conjugate gradient method. We shall show that the estimator achieves the (nearly) optimal convergence rate on a class of ellipsoids and we introduce an early stopping rule which adapts to the unknown degree of ill-posedness. Some theoretical and simulation comparison between the estimator and the principal component regression estimator is provided.
    

