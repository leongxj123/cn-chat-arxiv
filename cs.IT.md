# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels](https://arxiv.org/abs/2403.07735) | HSIC估计的极小化率对平移不变核的独立性度量具有重要意义 |
| [^2] | [Large Language Model Evaluation via Matrix Entropy](https://arxiv.org/abs/2401.17139) | 本文引入了矩阵熵，一种基于信息论和几何原理的新型指标，用于评估大型语言模型（LLMs）的数据压缩能力。该指标反映了模型提取相关信息和消除不必要元素的能力，为评估语言模型的固有能力提供了洞察。在单模态和多模态设置中都展示了其适用性。 |
| [^3] | [Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA.](http://arxiv.org/abs/2303.06198) | 本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。 |

# 详细

[^1]: HSIC估计的极小化率对平移不变核

    The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels

    [https://arxiv.org/abs/2403.07735](https://arxiv.org/abs/2403.07735)

    HSIC估计的极小化率对平移不变核的独立性度量具有重要意义

    

    Kernel技术是数据科学和统计学中最有影响力的方法之一。在温和条件下，与核相关的再生核希尔伯特空间能够编码$M\ge 2$个随机变量的独立性。在核上依赖的最普遍的独立性度量可能是所谓的Hilbert-Schmidt独立性准则(HSIC; 在统计文献中也称为距离协方差)。尽管自近二十年前引入以来已经有各种现有的设计的HSIC估计量，HSIC可以被估计的速度的基本问题仍然是开放的。在这项工作中，我们证明了对于包含具有连续有界平移不变特征核的高斯Borel测度在$\mathbb R^d$上的HSIC估计的极小化最优速率是$\mathcal O\!\left(n^{-1/2}\right)$。具体地，我们的结果意味着许多方面在极小化意义上的最优性

    arXiv:2403.07735v1 Announce Type: cross  Abstract: Kernel techniques are among the most influential approaches in data science and statistics. Under mild conditions, the reproducing kernel Hilbert space associated to a kernel is capable of encoding the independence of $M\ge 2$ random variables. Probably the most widespread independence measure relying on kernels is the so-called Hilbert-Schmidt independence criterion (HSIC; also referred to as distance covariance in the statistics literature). Despite various existing HSIC estimators designed since its introduction close to two decades ago, the fundamental question of the rate at which HSIC can be estimated is still open. In this work, we prove that the minimax optimal rate of HSIC estimation on $\mathbb R^d$ for Borel measures containing the Gaussians with continuous bounded translation-invariant characteristic kernels is $\mathcal O\!\left(n^{-1/2}\right)$. Specifically, our result implies the optimality in the minimax sense of many 
    
[^2]: 通过矩阵熵评估大型语言模型

    Large Language Model Evaluation via Matrix Entropy

    [https://arxiv.org/abs/2401.17139](https://arxiv.org/abs/2401.17139)

    本文引入了矩阵熵，一种基于信息论和几何原理的新型指标，用于评估大型语言模型（LLMs）的数据压缩能力。该指标反映了模型提取相关信息和消除不必要元素的能力，为评估语言模型的固有能力提供了洞察。在单模态和多模态设置中都展示了其适用性。

    

    大型语言模型（LLMs）通过将强大的能力扩展到多模态领域，使自然语言处理领域发生了革命。因此，为LLMs定义适当且多样化的评估指标至关重要。在本文中，我们引入了矩阵熵，一种根植于信息论和几何原理的新型指标，用于量化LLMs中的数据压缩能力。它反映了模型提取相关信息和消除不必要元素的能力，从而提供了对语言模型固有能力的洞察。具体而言，我们展示了它在单模态（语言）和多模态设置中的适用性。对于语言模型，我们的发现揭示出表示的矩阵熵在模型扩大时遵循一个缩放定律类型的降低，这作为传统损失缩放定律的补充。对于多模态设置，我们还提出了一种基于矩阵熵的评估方法，用于评估一个模型的性能。

    Large language models (LLMs) have revolutionized the field of natural language processing, extending their strong capabilities into multi-modal domains. Thus, it is vital to define proper and diversified metrics for the evaluation of LLMs.   In this paper, we introduce matrix entropy, a novel metric rooted in information theory and geometry principles to quantify the data compression proficiency in LLMs. It reflects the model's ability to extract relevant information and eliminate unnecessary elements, thereby providing insight into the language model's intrinsic capability. Specifically, we demonstrate its applicability in both single-modal (language) and multi-modal settings. For language models, our findings reveal that the matrix entropy of representations follows a scaling law type reduction when the model scales up, serving as a complement to the traditional loss scaling law. For the multi-modal setting, we also propose an evaluation method based on matrix entropy for assessing a
    
[^3]: 克服异方差PCA中病态问题的缩减算法

    Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA. (arXiv:2303.06198v1 [math.ST])

    [http://arxiv.org/abs/2303.06198](http://arxiv.org/abs/2303.06198)

    本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。

    This paper proposes a novel algorithm, called Deflated-HeteroPCA, that overcomes the curse of ill-conditioning in heteroskedastic PCA while achieving near-optimal and condition-number-free theoretical guarantees.

    本文关注于从受污染的数据中估计低秩矩阵X*的列子空间。当存在异方差噪声和不平衡的维度（即n2 >> n1）时，如何在容纳最广泛的信噪比范围的同时获得最佳的统计精度变得特别具有挑战性。虽然最先进的算法HeteroPCA成为解决这个问题的强有力的解决方案，但它遭受了“病态问题的诅咒”，即随着X*的条件数增长，其性能会下降。为了克服这个关键问题而不影响允许的信噪比范围，我们提出了一种新的算法，称为缩减异方差PCA，它在$\ell_2$和$\ell_{2,\infty}$统计精度方面实现了近乎最优和无条件数的理论保证。所提出的算法将谱分成两部分

    This paper is concerned with estimating the column subspace of a low-rank matrix $\boldsymbol{X}^\star \in \mathbb{R}^{n_1\times n_2}$ from contaminated data. How to obtain optimal statistical accuracy while accommodating the widest range of signal-to-noise ratios (SNRs) becomes particularly challenging in the presence of heteroskedastic noise and unbalanced dimensionality (i.e., $n_2\gg n_1$). While the state-of-the-art algorithm $\textsf{HeteroPCA}$ emerges as a powerful solution for solving this problem, it suffers from "the curse of ill-conditioning," namely, its performance degrades as the condition number of $\boldsymbol{X}^\star$ grows. In order to overcome this critical issue without compromising the range of allowable SNRs, we propose a novel algorithm, called $\textsf{Deflated-HeteroPCA}$, that achieves near-optimal and condition-number-free theoretical guarantees in terms of both $\ell_2$ and $\ell_{2,\infty}$ statistical accuracy. The proposed algorithm divides the spectrum
    

