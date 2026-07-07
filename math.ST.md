# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^2] | [Learning linear dynamical systems under convex constraints.](http://arxiv.org/abs/2303.15121) | 本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。 |

# 详细

[^1]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^2]: 在凸约束下学习线性动态系统

    Learning linear dynamical systems under convex constraints. (arXiv:2303.15121v1 [math.ST])

    [http://arxiv.org/abs/2303.15121](http://arxiv.org/abs/2303.15121)

    本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。

    

    我们考虑从单个轨迹中识别线性动态系统的问题。最近的研究主要关注未对系统矩阵 $A^* \in \mathbb{R}^{n \times n}$ 进行结构假设的情况，并对普通最小二乘 (OLS) 估计器进行了详细分析。我们假设可用先前的 $A^*$ 的结构信息，可以在包含 $A^*$ 的凸集 $\mathcal{K}$ 中捕获。对于随后的受约束最小二乘估计的解，我们推导出 Frobenius 范数下依赖于 $\mathcal{K}$ 在 $A^*$ 处切锥的局部大小的非渐进误差界。为了说明这一结果的有用性，我们将其实例化为以下设置：(i) $\mathcal{K}$ 是 $\mathbb{R}^{n \times n}$ 中的 $d$ 维子空间，或者 (ii) $A^*$ 是 $k$ 稀疏的，$\mathcal{K}$ 是适当缩放的 $\ell_1$ 球。在 $d, k \ll n^2$ 的区域中，我们的误差界对于相同的统计和噪声假设比 OLS 估计器获得了改进。

    We consider the problem of identification of linear dynamical systems from a single trajectory. Recent results have predominantly focused on the setup where no structural assumption is made on the system matrix $A^* \in \mathbb{R}^{n \times n}$, and have consequently analyzed the ordinary least squares (OLS) estimator in detail. We assume prior structural information on $A^*$ is available, which can be captured in the form of a convex set $\mathcal{K}$ containing $A^*$. For the solution of the ensuing constrained least squares estimator, we derive non-asymptotic error bounds in the Frobenius norm which depend on the local size of the tangent cone of $\mathcal{K}$ at $A^*$. To illustrate the usefulness of this result, we instantiate it for the settings where, (i) $\mathcal{K}$ is a $d$ dimensional subspace of $\mathbb{R}^{n \times n}$, or (ii) $A^*$ is $k$-sparse and $\mathcal{K}$ is a suitably scaled $\ell_1$ ball. In the regimes where $d, k \ll n^2$, our bounds improve upon those obta
    

