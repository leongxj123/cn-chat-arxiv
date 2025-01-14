# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Measure transfer via stochastic slicing and matching.](http://arxiv.org/abs/2307.05705) | 本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。 |
| [^2] | [Symmetry & Critical Points for Symmetric Tensor Decompositions Problems.](http://arxiv.org/abs/2306.07886) | 本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。 |

# 详细

[^1]: 通过随机切片和配准进行测量转移

    Measure transfer via stochastic slicing and matching. (arXiv:2307.05705v1 [math.NA])

    [http://arxiv.org/abs/2307.05705](http://arxiv.org/abs/2307.05705)

    本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。

    

    本论文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案。类似于切片Wasserstein距离，这些方案受益于一维最优输运问题的闭式解的可用性和相关计算优势。尽管这些方案已经在数据科学应用中取得了成功，但关于它们的收敛性的结果不太多。本文的主要贡献是对随机切片和配准方案提供了几乎必然收敛的证明。该证明建立在将其解释为Wasserstein空间上的随机梯度下降方案的基础之上。同时还展示了关于逐步图像变形的数值示例。

    This paper studies iterative schemes for measure transfer and approximation problems, which are defined through a slicing-and-matching procedure. Similar to the sliced Wasserstein distance, these schemes benefit from the availability of closed-form solutions for the one-dimensional optimal transport problem and the associated computational advantages. While such schemes have already been successfully utilized in data science applications, not too many results on their convergence are available. The main contribution of this paper is an almost sure convergence proof for stochastic slicing-and-matching schemes. The proof builds on an interpretation as a stochastic gradient descent scheme on the Wasserstein space. Numerical examples on step-wise image morphing are demonstrated as well.
    
[^2]: 对称张量分解问题的对称性与临界点

    Symmetry & Critical Points for Symmetric Tensor Decompositions Problems. (arXiv:2306.07886v1 [math.OC])

    [http://arxiv.org/abs/2306.07886](http://arxiv.org/abs/2306.07886)

    本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。

    

    本文考虑了将一个实对称张量分解成秩为1项之和的非凸优化问题。利用其丰富的对称结构，导出Puiseux级数表示的一系列临界点，并获得了关于临界值和Hessian谱的精确分析估计。这些结果揭示了各种几何障碍，阻碍了局部优化方法的使用，最后，利用一个牛顿多面体论证了固定对称性的所有临界点的完全枚举，并证明了与全局最小值的集合相比，由于对称性的存在，临界点的集合可能会显示出组合的丰富性。

    We consider the non-convex optimization problem associated with the decomposition of a real symmetric tensor into a sum of rank one terms. Use is made of the rich symmetry structure to derive Puiseux series representations of families of critical points, and so obtain precise analytic estimates on the critical values and the Hessian spectrum. The sharp results make possible an analytic characterization of various geometric obstructions to local optimization methods, revealing in particular a complex array of saddles and local minima which differ by their symmetry, structure and analytic properties. A desirable phenomenon, occurring for all critical points considered, concerns the index of a point, i.e., the number of negative Hessian eigenvalues, increasing with the value of the objective function. Lastly, a Newton polytope argument is used to give a complete enumeration of all critical points of fixed symmetry, and it is shown that contrarily to the set of global minima which remains 
    

