# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions](https://arxiv.org/abs/2402.16442) | 本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。 |

# 详细

[^1]: 在具有配对次模模函数的分布式大于内存的子集选择问题研究

    On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions

    [https://arxiv.org/abs/2402.16442](https://arxiv.org/abs/2402.16442)

    本文提出了一种新颖的分布式约束算法，通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    

    许多学习问题取决于子集选择的基本问题，即确定一组重要和代表性的点。本文提出了一种具有可证估计近似保证的新颖分布式约束算法，它通过迭代绑定最小和最大效用值来选择高质量的点并丢弃不重要的点。

    arXiv:2402.16442v1 Announce Type: cross  Abstract: Many learning problems hinge on the fundamental problem of subset selection, i.e., identifying a subset of important and representative points. For example, selecting the most significant samples in ML training cannot only reduce training costs but also enhance model quality. Submodularity, a discrete analogue of convexity, is commonly used for solving subset selection problems. However, existing algorithms for optimizing submodular functions are sequential, and the prior distributed methods require at least one central machine to fit the target subset. In this paper, we relax the requirement of having a central machine for the target subset by proposing a novel distributed bounding algorithm with provable approximation guarantees. The algorithm iteratively bounds the minimum and maximum utility values to select high quality points and discard the unimportant ones. When bounding does not find the complete subset, we use a multi-round, 
    

