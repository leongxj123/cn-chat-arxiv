# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved Convergence Rates of Windowed Anderson Acceleration for Symmetric Fixed-Point Iterations](https://arxiv.org/abs/2311.02490) | 窗口式安德森加速在对称不动点迭代中具有改进的根线性收敛率，模拟和实验结果证实其超越标准不动点方法。 |
| [^2] | [Weighted least-squares approximation with determinantal point processes and generalized volume sampling.](http://arxiv.org/abs/2312.14057) | 该论文研究了使用行列式点过程和广义体积取样进行加权最小二乘逼近的问题，提出了广义版本的体积标准化取样算法，并证明了该算法在期望上的准最优性以及在某些规范向量空间中的逼近结果。 |

# 详细

[^1]: 对称不动点迭代的窗口式安德森加速收敛率的改进

    Improved Convergence Rates of Windowed Anderson Acceleration for Symmetric Fixed-Point Iterations

    [https://arxiv.org/abs/2311.02490](https://arxiv.org/abs/2311.02490)

    窗口式安德森加速在对称不动点迭代中具有改进的根线性收敛率，模拟和实验结果证实其超越标准不动点方法。

    

    本文研究了常用的窗口式安德森加速（AA）算法用于不动点方法，$x^{(k+1)}=q(x^{(k)})$。它首次证明了当算子$q$是线性且对称时，使用先前迭代的滑动窗口的窗口式AA算法能够改进根线性收敛因子，超过不动点迭代。当$q$是非线性的，但在固定点处具有对称雅可比矩阵时，经过略微修改的AA算法被证明对比不动点迭代具有类似的根线性收敛因子改进。模拟验证了我们的观察。此外，使用不同数据模型进行的实验表明，在Tyler的M估计中，AA明显优于标准的不动点方法。

    arXiv:2311.02490v2 Announce Type: replace-cross  Abstract: This paper studies the commonly utilized windowed Anderson acceleration (AA) algorithm for fixed-point methods, $x^{(k+1)}=q(x^{(k)})$. It provides the first proof that when the operator $q$ is linear and symmetric the windowed AA, which uses a sliding window of prior iterates, improves the root-linear convergence factor over the fixed-point iterations. When $q$ is nonlinear, yet has a symmetric Jacobian at a fixed point, a slightly modified AA algorithm is proved to have an analogous root-linear convergence factor improvement over fixed-point iterations. Simulations verify our observations. Furthermore, experiments with different data models demonstrate AA is significantly superior to the standard fixed-point methods for Tyler's M-estimation.
    
[^2]: 基于行列式点过程和广义体积取样的加权最小二乘逼近

    Weighted least-squares approximation with determinantal point processes and generalized volume sampling. (arXiv:2312.14057v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2312.14057](http://arxiv.org/abs/2312.14057)

    该论文研究了使用行列式点过程和广义体积取样进行加权最小二乘逼近的问题，提出了广义版本的体积标准化取样算法，并证明了该算法在期望上的准最优性以及在某些规范向量空间中的逼近结果。

    

    我们考虑使用给定的m维空间V_m中的元素，借助于一些特征映射φ，通过对随机点x_1，...，x_n处的函数进行评估，来逼近函数从L^2到函数。在回顾一些关于使用独立同分布点的最优加权最小二乘的结果之后，我们考虑使用投影行列式点过程（DPP）或体积取样的加权最小二乘。这些分布在选定的特征φ(x_i)中引入了点之间的依赖性，以促进多样性。我们首先提供了广义版本的体积标准化取样，使用样本数n = O(mlog(m))得到了期望上的准最优结果，这意味着期望的L^2误差受到一个常数乘以在L^2中的最佳逼近误差的限制。此外，进一步假设函数在某个嵌入在L^2中的规范向量空间H中，我们进一步证明了逼近的结果。

    We consider the problem of approximating a function from $L^2$ by an element of a given $m$-dimensional space $V_m$, associated with some feature map $\varphi$, using evaluations of the function at random points $x_1,\dots,x_n$. After recalling some results on optimal weighted least-squares using independent and identically distributed points, we consider weighted least-squares using projection determinantal point processes (DPP) or volume sampling. These distributions introduce dependence between the points that promotes diversity in the selected features $\varphi(x_i)$. We first provide a generalized version of volume-rescaled sampling yielding quasi-optimality results in expectation with a number of samples $n = O(m\log(m))$, that means that the expected $L^2$ error is bounded by a constant times the best approximation error in $L^2$. Also, further assuming that the function is in some normed vector space $H$ continuously embedded in $L^2$, we further prove that the approximation is
    

