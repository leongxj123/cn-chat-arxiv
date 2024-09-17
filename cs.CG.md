# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parameterized Approximation for Robust Clustering in Discrete Geometric Spaces.](http://arxiv.org/abs/2305.07316) | 本文研究了Robust $(k, z)$-Clustering问题，提出了在离散几何空间中的参数化近似解法，可以在多项式时间内获得$O(\log m/\log\log m)$的近似因子，在FPT时间内可以获得$(3^z+\epsilon)$的近似因子。 |

# 详细

[^1]: 离散几何空间中抗干扰聚类问题的参数化近似

    Parameterized Approximation for Robust Clustering in Discrete Geometric Spaces. (arXiv:2305.07316v1 [cs.DS])

    [http://arxiv.org/abs/2305.07316](http://arxiv.org/abs/2305.07316)

    本文研究了Robust $(k, z)$-Clustering问题，提出了在离散几何空间中的参数化近似解法，可以在多项式时间内获得$O(\log m/\log\log m)$的近似因子，在FPT时间内可以获得$(3^z+\epsilon)$的近似因子。

    

    本文研究了Robust $(k, z)$-Clustering问题，该问题出现在鲁棒优化和算法公平性的领域中。已知在多项式时间内该问题具有$O(\log m/\log\log m)$的近似因子，在FPT时间内具有$(3^z+\epsilon)$的近似算法。

    We consider the well-studied Robust $(k, z)$-Clustering problem, which generalizes the classic $k$-Median, $k$-Means, and $k$-Center problems. Given a constant $z\ge 1$, the input to Robust $(k, z)$-Clustering is a set $P$ of $n$ weighted points in a metric space $(M,\delta)$ and a positive integer $k$. Further, each point belongs to one (or more) of the $m$ many different groups $S_1,S_2,\ldots,S_m$. Our goal is to find a set $X$ of $k$ centers such that $\max_{i \in [m]} \sum_{p \in S_i} w(p) \delta(p,X)^z$ is minimized.  This problem arises in the domains of robust optimization [Anthony, Goyal, Gupta, Nagarajan, Math. Oper. Res. 2010] and in algorithmic fairness. For polynomial time computation, an approximation factor of $O(\log m/\log\log m)$ is known [Makarychev, Vakilian, COLT $2021$], which is tight under a plausible complexity assumption even in the line metrics. For FPT time, there is a $(3^z+\epsilon)$-approximation algorithm, which is tight under GAP-ETH [Goyal, Jaiswal, In
    

