# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [Inference for Synthetic Controls via Refined Placebo Tests.](http://arxiv.org/abs/2401.07152) | 本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。 |
| [^3] | [Semi-Supervised Causal Inference: Generalizable and Double Robust Inference for Average Treatment Effects under Selection Bias with Decaying Overlap.](http://arxiv.org/abs/2305.12789) | 本论文提出了一种新的针对高维情况下缺失标签且存在选择偏差的平均处理效应估计方法，它具有良好的一致性和渐近正态性。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: 通过精细安慰剂测试进行合成控制的推断

    Inference for Synthetic Controls via Refined Placebo Tests. (arXiv:2401.07152v1 [stat.ME])

    [http://arxiv.org/abs/2401.07152](http://arxiv.org/abs/2401.07152)

    本文提出了一种通过精细安慰剂测试进行合成控制的推断方法，用于解决只有一个处理单元和少数对照单元的问题，并解决了样本量小和实际估计过程简化的问题。

    

    合成控制方法通常用于只有一个处理单元和少数对照单元的问题。在这种情况下，一种常见的推断任务是测试关于对待处理单元的平均处理效应的零假设。由于（1）样本量较小导致大样本近似不稳定和（2）在实践中实施的估计过程的简化，因此通常无法满足渐近合理性的推断程序常常不令人满意。一种替代方法是置换推断，它与常见的称为安慰剂测试的诊断相关。当治疗均匀分配时，它在有限样本中具有可证明的 Type-I 错误保证，而无需简化方法。尽管具有这种健壮性，安慰剂测试由于只从 $N$ 个参考估计构造零分布，其中 $N$ 是样本量，因此分辨率较低。这在常见的水平 $\alpha = 0.05$ 的统计推断中形成了一个障碍，特别是在小样本问题中。

    The synthetic control method is often applied to problems with one treated unit and a small number of control units. A common inferential task in this setting is to test null hypotheses regarding the average treatment effect on the treated. Inference procedures that are justified asymptotically are often unsatisfactory due to (1) small sample sizes that render large-sample approximation fragile and (2) simplification of the estimation procedure that is implemented in practice. An alternative is permutation inference, which is related to a common diagnostic called the placebo test. It has provable Type-I error guarantees in finite samples without simplification of the method, when the treatment is uniformly assigned. Despite this robustness, the placebo test suffers from low resolution since the null distribution is constructed from only $N$ reference estimates, where $N$ is the sample size. This creates a barrier for statistical inference at a common level like $\alpha = 0.05$, especia
    
[^3]: 半监督因果推断：面向衰减重叠的选择偏差下可泛化的双稳估计平均处理效应

    Semi-Supervised Causal Inference: Generalizable and Double Robust Inference for Average Treatment Effects under Selection Bias with Decaying Overlap. (arXiv:2305.12789v1 [stat.ME])

    [http://arxiv.org/abs/2305.12789](http://arxiv.org/abs/2305.12789)

    本论文提出了一种新的针对高维情况下缺失标签且存在选择偏差的平均处理效应估计方法，它具有良好的一致性和渐近正态性。

    

    平均处理效应（ATE）估计是因果推断文献中的一个重要问题，尤其是在高维混淆变量的情况下受到了极大的关注。本文中，我们考虑了在高维情况下存在可能缺失的标签情况下的ATE估计问题。标记指示符的条件倾向得分允许依赖于协变量，并且随着样本大小的衰减而衰减——从而允许未标记数据大小比标记数据大小增长得更快。这种情况填补了半监督（SS）和缺失数据文献中的重要空白。我们考虑了允许选择偏差的随机缺失（MAR）机制——这通常在标准的SS文献中是禁止的，并且在缺失数据文献中通常需要一个正性条件。我们首先提出了一种针对ATE的一般双稳DR-DMAR（decaying）SS估计器，这种估计器具有良好的一致性和渐近正态性。

    Average treatment effect (ATE) estimation is an essential problem in the causal inference literature, which has received significant recent attention, especially with the presence of high-dimensional confounders. We consider the ATE estimation problem in high dimensions when the observed outcome (or label) itself is possibly missing. The labeling indicator's conditional propensity score is allowed to depend on the covariates, and also decay uniformly with sample size - thus allowing for the unlabeled data size to grow faster than the labeled data size. Such a setting fills in an important gap in both the semi-supervised (SS) and missing data literatures. We consider a missing at random (MAR) mechanism that allows selection bias - this is typically forbidden in the standard SS literature, and without a positivity condition - this is typically required in the missing data literature. We first propose a general doubly robust 'decaying' MAR (DR-DMAR) SS estimator for the ATE, which is cons
    

