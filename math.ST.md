# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel.](http://arxiv.org/abs/2401.02520) | 本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。 |

# 详细

[^1]: 在任意元素间依赖下的结构化矩阵学习与马尔可夫转移核估计

    Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel. (arXiv:2401.02520v1 [stat.ML])

    [http://arxiv.org/abs/2401.02520](http://arxiv.org/abs/2401.02520)

    本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。

    

    结构化矩阵估计问题通常在强噪声依赖假设下进行研究。本文考虑噪声低秩加稀疏矩阵恢复的一般框架，其中噪声矩阵可以来自任意具有元素间任意依赖的联合分布。我们提出了一个无关相位约束的最小二乘估计器，并且证明了它在各种噪声分布下都是紧致的，既满足确定性下界又匹配最小化风险。为了实现这一点，我们建立了一个新颖的结果，断言两个任意的低秩无关矩阵之间的差异必须在其元素上扩散能量，换句话说不能太稀疏，这揭示了无关低秩矩阵的结构，可能引起独立兴趣。然后，我们展示了我们框架在几个重要的统计机器学习问题中的应用。在估计结构化马尔可夫转移核的问题中，采用了这种方法。

    The problem of structured matrix estimation has been studied mostly under strong noise dependence assumptions. This paper considers a general framework of noisy low-rank-plus-sparse matrix recovery, where the noise matrix may come from any joint distribution with arbitrary dependence across entries. We propose an incoherent-constrained least-square estimator and prove its tightness both in the sense of deterministic lower bound and matching minimax risks under various noise distributions. To attain this, we establish a novel result asserting that the difference between two arbitrary low-rank incoherent matrices must spread energy out across its entries, in other words cannot be too sparse, which sheds light on the structure of incoherent low-rank matrices and may be of independent interest. We then showcase the applications of our framework to several important statistical machine learning problems. In the problem of estimating a structured Markov transition kernel, the proposed method
    

