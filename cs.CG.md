# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [For Kernel Range Spaces a Constant Number of Queries Are Sufficient.](http://arxiv.org/abs/2306.16516) | 对于核范围空间，引入了ε-覆盖的概念，用于处理不确定或不精确的数据分析任务。 |

# 详细

[^1]: 对于核范围空间，只需要固定数量的查询就足够了

    For Kernel Range Spaces a Constant Number of Queries Are Sufficient. (arXiv:2306.16516v1 [cs.CG])

    [http://arxiv.org/abs/2306.16516](http://arxiv.org/abs/2306.16516)

    对于核范围空间，引入了ε-覆盖的概念，用于处理不确定或不精确的数据分析任务。

    

    我们引入了核范围空间的ε-覆盖概念。核范围空间涉及一个点集X⊂R^d和由固定核函数（例如高斯核函数K(p,·)=exp(-||p-·||^2)）定义的查询空间。对于大小为n的点集X，查询返回一个值向量Rp∈R^n，其中第i个坐标(Rp)_i=K(p,x_i)，其中x_i∈X。ε-覆盖是点集Q⊂R^d的子集，对于任意p∈R^d，存在q∈Q使得||(Rp-Rq)/n||_1≤ε。这是Haussler在组合范围空间（例如由球查询定义的点集子集）中ε-覆盖概念的平滑模拟，其中得到的向量Rp是{0,1}^n而不是[0,1]^n。这些范围空间的核版本出现在数据分析任务中，其中坐标可能是不确定或不精确的，因此希望在范围查询中添加一些灵活性。

    We introduce the notion of an $\varepsilon$-cover for a kernel range space. A kernel range space concerns a set of points $X \subset \mathbb{R}^d$ and the space of all queries by a fixed kernel (e.g., a Gaussian kernel $K(p,\cdot) = \exp(-\|p-\cdot\|^2)$). For a point set $X$ of size $n$, a query returns a vector of values $R_p \in \mathbb{R}^n$, where the $i$th coordinate $(R_p)_i = K(p,x_i)$ for $x_i \in X$. An $\varepsilon$-cover is a subset of points $Q \subset \mathbb{R}^d$ so for any $p \in \mathbb{R}^d$ that $\frac{1}{n} \|R_p R_q\|_1\leq \varepsilon$ for some $q \in Q$. This is a smooth analog of Haussler's notion of $\varepsilon$-covers for combinatorial range spaces (e.g., defined by subsets of points within a ball query) where the resulting vectors $R_p$ are in $\{0,1\}^n$ instead of $[0,1]^n$. The kernel versions of these range spaces show up in data analysis tasks where the coordinates may be uncertain or imprecise, and hence one wishes to add some flexibility in the not
    

