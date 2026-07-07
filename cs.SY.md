# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning linear dynamical systems under convex constraints.](http://arxiv.org/abs/2303.15121) | 本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。 |

# 详细

[^1]: 在凸约束下学习线性动态系统

    Learning linear dynamical systems under convex constraints. (arXiv:2303.15121v1 [math.ST])

    [http://arxiv.org/abs/2303.15121](http://arxiv.org/abs/2303.15121)

    本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。

    

    我们考虑从单个轨迹中识别线性动态系统的问题。最近的研究主要关注未对系统矩阵 $A^* \in \mathbb{R}^{n \times n}$ 进行结构假设的情况，并对普通最小二乘 (OLS) 估计器进行了详细分析。我们假设可用先前的 $A^*$ 的结构信息，可以在包含 $A^*$ 的凸集 $\mathcal{K}$ 中捕获。对于随后的受约束最小二乘估计的解，我们推导出 Frobenius 范数下依赖于 $\mathcal{K}$ 在 $A^*$ 处切锥的局部大小的非渐进误差界。为了说明这一结果的有用性，我们将其实例化为以下设置：(i) $\mathcal{K}$ 是 $\mathbb{R}^{n \times n}$ 中的 $d$ 维子空间，或者 (ii) $A^*$ 是 $k$ 稀疏的，$\mathcal{K}$ 是适当缩放的 $\ell_1$ 球。在 $d, k \ll n^2$ 的区域中，我们的误差界对于相同的统计和噪声假设比 OLS 估计器获得了改进。

    We consider the problem of identification of linear dynamical systems from a single trajectory. Recent results have predominantly focused on the setup where no structural assumption is made on the system matrix $A^* \in \mathbb{R}^{n \times n}$, and have consequently analyzed the ordinary least squares (OLS) estimator in detail. We assume prior structural information on $A^*$ is available, which can be captured in the form of a convex set $\mathcal{K}$ containing $A^*$. For the solution of the ensuing constrained least squares estimator, we derive non-asymptotic error bounds in the Frobenius norm which depend on the local size of the tangent cone of $\mathcal{K}$ at $A^*$. To illustrate the usefulness of this result, we instantiate it for the settings where, (i) $\mathcal{K}$ is a $d$ dimensional subspace of $\mathbb{R}^{n \times n}$, or (ii) $A^*$ is $k$-sparse and $\mathcal{K}$ is a suitably scaled $\ell_1$ ball. In the regimes where $d, k \ll n^2$, our bounds improve upon those obta
    

