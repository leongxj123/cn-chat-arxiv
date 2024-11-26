# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression](https://arxiv.org/abs/2403.13565) | 提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。 |
| [^2] | [Selecting informative conformal prediction sets with false coverage rate control](https://arxiv.org/abs/2403.12295) | 提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。 |
| [^3] | [Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay.](http://arxiv.org/abs/2401.01599) | 本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。 |
| [^4] | [Estimating the roughness exponent of stochastic volatility from discrete observations of the realized variance.](http://arxiv.org/abs/2307.02582) | 本文提出了一种新的估计器，用于从离散观测中测量连续轨迹的粗糙指数，该估计器适用于随机波动模型的估计，并在大多数分数布朗运动样本路径中收敛。 |
| [^5] | [Sample Complexity of Probability Divergences under Group Symmetry.](http://arxiv.org/abs/2302.01915) | 本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。 |

# 详细

[^1]: AdaTrans：针对高维回归的特征自适应与样本自适应迁移学习

    AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression

    [https://arxiv.org/abs/2403.13565](https://arxiv.org/abs/2403.13565)

    提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。

    

    我们考虑高维背景下的迁移学习问题，在该问题中，特征维度大于样本大小。为了学习可迁移的信息，该信息可能在特征或源样本之间变化，我们提出一种自适应迁移学习方法，可以检测和聚合特征-wise (F-AdaTrans)或样本-wise (S-AdaTrans)可迁移结构。我们通过采用一种新颖的融合惩罚方法，结合权重，可以根据可迁移结构进行调整。为了选择权重，我们提出了一个在理论上建立，数据驱动的过程，使得 F-AdaTrans 能够选择性地将可迁移的信号与目标融合在一起，同时滤除非可迁移的信号，S-AdaTrans则可以获得每个源样本传递的信息的最佳组合。我们建立了非渐近速率，可以在特殊情况下恢复现有的近最小似乎最优速率。效果证明...

    arXiv:2403.13565v1 Announce Type: cross  Abstract: We consider the transfer learning problem in the high dimensional setting, where the feature dimension is larger than the sample size. To learn transferable information, which may vary across features or the source samples, we propose an adaptive transfer learning method that can detect and aggregate the feature-wise (F-AdaTrans) or sample-wise (S-AdaTrans) transferable structures. We achieve this by employing a novel fused-penalty, coupled with weights that can adapt according to the transferable structure. To choose the weight, we propose a theoretically informed, data-driven procedure, enabling F-AdaTrans to selectively fuse the transferable signals with the target while filtering out non-transferable signals, and S-AdaTrans to obtain the optimal combination of information transferred from each source sample. The non-asymptotic rates are established, which recover existing near-minimax optimal rates in special cases. The effectivene
    
[^2]: 通过控制虚警覆盖率选择信息量丰富的符合预测集

    Selecting informative conformal prediction sets with false coverage rate control

    [https://arxiv.org/abs/2403.12295](https://arxiv.org/abs/2403.12295)

    提出了一种新的统一框架，用于构建信息丰富的符合预测集，同时控制所选样本的虚警覆盖率。

    

    在监督学习中，包括回归和分类，符合方法为任何机器学习预测器提供预测结果/标签的预测集合，具有有限样本覆盖率。在这里我们考虑了这样一种情况，即这种预测集合是经过选择过程得到的。该选择过程要求选择的预测集在某种明确定义的意义上是“信息量丰富的”。我们考虑了分类和回归设置，在这些设置中，分析人员可能只考虑具有预测标签集或预测区间足够小、不包括空值或遵守其他适当的“单调”约束的样本为具有信息量丰富的。虽然这涵盖了各种应用中可能感兴趣的许多设置，我们开发了一个统一的框架，用来构建这样的信息量丰富的符合预测集，同时控制所选样本上的虚警覆盖率（FCR）。

    arXiv:2403.12295v1 Announce Type: cross  Abstract: In supervised learning, including regression and classification, conformal methods provide prediction sets for the outcome/label with finite sample coverage for any machine learning predictors. We consider here the case where such prediction sets come after a selection process. The selection process requires that the selected prediction sets be `informative' in a well defined sense. We consider both the classification and regression settings where the analyst may consider as informative only the sample with prediction label sets or prediction intervals small enough, excluding null values, or obeying other appropriate `monotone' constraints. While this covers many settings of possible interest in various applications, we develop a unified framework for building such informative conformal prediction sets while controlling the false coverage rate (FCR) on the selected sample. While conformal prediction sets after selection have been the f
    
[^3]: 分析谱算法在幂律衰减下的泛化误差曲线

    Generalization Error Curves for Analytic Spectral Algorithms under Power-law Decay. (arXiv:2401.01599v1 [cs.LG])

    [http://arxiv.org/abs/2401.01599](http://arxiv.org/abs/2401.01599)

    本文研究了核回归方法的泛化误差曲线，对核梯度下降方法和其他分析谱算法在核回归中的泛化误差进行了全面特征化，从而提高了对训练宽神经网络泛化行为的理解，并提出了一种新的技术贡献-分析功能论证。

    

    某些核回归方法的泛化误差曲线旨在确定在不同源条件、噪声水平和正则化参数选择下的泛化误差的确切顺序，而不是最小化率。在本文中，在温和的假设下，我们严格给出了核梯度下降方法（以及大类分析谱算法）在核回归中的泛化误差曲线的完整特征化。因此，我们可以提高核插值的近不一致性，并澄清具有更高资格的核回归算法的饱和效应，等等。由于神经切线核理论的帮助，这些结果极大地提高了我们对训练宽神经网络的泛化行为的理解。一种新颖的技术贡献，即分析功能论证，可能具有独立的兴趣。

    The generalization error curve of certain kernel regression method aims at determining the exact order of generalization error with various source condition, noise level and choice of the regularization parameter rather than the minimax rate. In this work, under mild assumptions, we rigorously provide a full characterization of the generalization error curves of the kernel gradient descent method (and a large class of analytic spectral algorithms) in kernel regression. Consequently, we could sharpen the near inconsistency of kernel interpolation and clarify the saturation effects of kernel regression algorithms with higher qualification, etc. Thanks to the neural tangent kernel theory, these results greatly improve our understanding of the generalization behavior of training the wide neural networks. A novel technical contribution, the analytic functional argument, might be of independent interest.
    
[^4]: 从离散实现方差的观测中估计随机波动的粗糙指数

    Estimating the roughness exponent of stochastic volatility from discrete observations of the realized variance. (arXiv:2307.02582v1 [q-fin.ST])

    [http://arxiv.org/abs/2307.02582](http://arxiv.org/abs/2307.02582)

    本文提出了一种新的估计器，用于从离散观测中测量连续轨迹的粗糙指数，该估计器适用于随机波动模型的估计，并在大多数分数布朗运动样本路径中收敛。

    

    本文考虑在随机波动模型中估计波动性的粗糙度，该模型是作为分数布朗运动（带漂移）的非线性函数而产生的。为此，我们引入一个新的估计量，该估计量测量连续轨迹的所谓粗糙指数，基于其原函数的离散观测。我们给出了对于基础轨迹的条件，在这些条件下，我们的估计器以严格路径方式收敛。然后我们验证了这些条件在几乎每个分数布朗运动（带漂移）样本路径中都得到满足。作为结果，在大类粗波动模型的背景下，我们得到了强一致性定理。数值模拟结果表明，在经过我们估计器的尺度不变修改后，我们的估计程序表现良好。

    We consider the problem of estimating the roughness of the volatility in a stochastic volatility model that arises as a nonlinear function of fractional Brownian motion with drift. To this end, we introduce a new estimator that measures the so-called roughness exponent of a continuous trajectory, based on discrete observations of its antiderivative. We provide conditions on the underlying trajectory under which our estimator converges in a strictly pathwise sense. Then we verify that these conditions are satisfied by almost every sample path of fractional Brownian motion (with drift). As a consequence, we obtain strong consistency theorems in the context of a large class of rough volatility models. Numerical simulations show that our estimation procedure performs well after passing to a scale-invariant modification of our estimator.
    
[^5]: 基于群对称性的概率差异的样本复杂度分析

    Sample Complexity of Probability Divergences under Group Symmetry. (arXiv:2302.01915v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.01915](http://arxiv.org/abs/2302.01915)

    本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。

    

    我们对于具有群不变性的分布变量在变分差异估计中的样本复杂度进行了严谨的量化分析。在Wasserstein-1距离和Lipschitz正则化α差异的情况下，样本复杂度的降低与群大小的维度相关。对于最大均值差异（MMD），样本复杂度的改进更加复杂，因为它不仅取决于群大小，还取决于内核的选择。 数值模拟验证了我们的理论。

    We rigorously quantify the improvement in the sample complexity of variational divergence estimations for group-invariant distributions. In the cases of the Wasserstein-1 metric and the Lipschitz-regularized $\alpha$-divergences, the reduction of sample complexity is proportional to an ambient-dimension-dependent power of the group size. For the maximum mean discrepancy (MMD), the improvement of sample complexity is more nuanced, as it depends on not only the group size but also the choice of kernel. Numerical simulations verify our theories.
    

