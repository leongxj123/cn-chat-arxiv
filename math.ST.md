# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes](https://arxiv.org/abs/2402.07296) | 该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。 |
| [^2] | [Sparse PCA With Multiple Components.](http://arxiv.org/abs/2209.14790) | 本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。 |

# 详细

[^1]: 估计几何遗传马尔可夫过程的混合系数

    Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes

    [https://arxiv.org/abs/2402.07296](https://arxiv.org/abs/2402.07296)

    该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。

    

    我们提出了一种方法来估计实值几何遗传马尔可夫过程的单个β-混合系数从一个单一的样本路径X0，X1，...，Xn。在对密度的标准光滑条件下，即对于每个m，对$(X_0,X_m)$对的联合密度都属于某个已知$s>0$的 Besov 空间$B^s_{1,\infty}(\mathbb R^2)$，我们得到了我们在这种情况下的估计器的预期误差的收敛速度为$\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ 的收敛速度。我们通过对估计误差的高概率界限进行了补充，并在状态空间有限的情况下获得了这些界限的类比。在这种情况下不需要密度的假设；预期误差率显示为$\mathcal O(\log(

    We propose methods to estimate the individual $\beta$-mixing coefficients of a real-valued geometrically ergodic Markov process from a single sample-path $X_0,X_1, \dots,X_n$. Under standard smoothness conditions on the densities, namely, that the joint density of the pair $(X_0,X_m)$ for each $m$ lies in a Besov space $B^s_{1,\infty}(\mathbb R^2)$ for some known $s>0$, we obtain a rate of convergence of order $\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ for the expected error of our estimator in this case\footnote{We use $[s]$ to denote the integer part of the decomposition $s=[s]+\{s\}$ of $s \in (0,\infty)$ into an integer term and a {\em strictly positive} remainder term $\{s\} \in (0,1]$.}. We complement this result with a high-probability bound on the estimation error, and further obtain analogues of these bounds in the case where the state-space is finite. Naturally no density assumptions are required in this setting; the expected error rate is shown to be of order $\mathcal O(\log(
    
[^2]: 多组分的稀疏主成分分析

    Sparse PCA With Multiple Components. (arXiv:2209.14790v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.14790](http://arxiv.org/abs/2209.14790)

    本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。

    

    稀疏主成分分析是一种用于以可解释的方式解释高维数据集方差的基本技术。这涉及解决一个稀疏性和正交性约束的凸最大化问题，其计算复杂度非常高。大多数现有的方法通过迭代计算一个稀疏主成分并缩减协方差矩阵来解决稀疏主成分分析，但在寻找多个相互正交的主成分时，这些方法不能保证所得解的正交性和最优性。我们挑战这种现状，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。此外，我们采用另一种方法来加强上界，我们使用额外的二阶锥不等式来加强上界。

    Sparse Principal Component Analysis (sPCA) is a cardinal technique for obtaining combinations of features, or principal components (PCs), that explain the variance of high-dimensional datasets in an interpretable manner. This involves solving a sparsity and orthogonality constrained convex maximization problem, which is extremely computationally challenging. Most existing works address sparse PCA via methods-such as iteratively computing one sparse PC and deflating the covariance matrix-that do not guarantee the orthogonality, let alone the optimality, of the resulting solution when we seek multiple mutually orthogonal PCs. We challenge this status by reformulating the orthogonality conditions as rank constraints and optimizing over the sparsity and rank constraints simultaneously. We design tight semidefinite relaxations to supply high-quality upper bounds, which we strengthen via additional second-order cone inequalities when each PC's individual sparsity is specified. Further, we de
    

