# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Swap $k$-Means++.](http://arxiv.org/abs/2309.16384) | 本论文通过多交换$k$-Means++算法的改进和扩展，提出了一种在$k$-means聚类问题中能够获得$9 + \varepsilon$近似比的局部搜索算法，并证明了该算法在实际中取得了显著的质量改进。 |

# 详细

[^1]: 多交换$k$-Means++算法

    Multi-Swap $k$-Means++. (arXiv:2309.16384v1 [cs.CG])

    [http://arxiv.org/abs/2309.16384](http://arxiv.org/abs/2309.16384)

    本论文通过多交换$k$-Means++算法的改进和扩展，提出了一种在$k$-means聚类问题中能够获得$9 + \varepsilon$近似比的局部搜索算法，并证明了该算法在实际中取得了显著的质量改进。

    

    Arthur和Vassilvitskii提出的$k$-means++算法通常被实践者选择用于优化流行的$k$-means聚类目标，并在期望中获得$O(\log k)$的近似度。为了获得更高质量的解，Lattanzi和Sohler提出了通过$k$-means++采样分布获得的$O(k \log \log k)$个局部搜索步骤的增强$k$-means++算法，从而得到$k$-means聚类问题的$c$近似解，其中$c$是一个较大的常数。在这里，我们通过考虑更大更复杂的局部搜索邻域来推广和扩展他们的局部搜索算法，从而可以同时交换多个中心。我们的算法实现了$9 + \varepsilon$的近似比，这是局部搜索可能的最佳结果。重要的是，我们证明了我们的方法在实际中取得了实质性的改进，我们显示出与Lattanzi和Sohler的方法相比的显著质量改进。

    The $k$-means++ algorithm of Arthur and Vassilvitskii (SODA 2007) is often the practitioners' choice algorithm for optimizing the popular $k$-means clustering objective and is known to give an $O(\log k)$-approximation in expectation. To obtain higher quality solutions, Lattanzi and Sohler (ICML 2019) proposed augmenting $k$-means++ with $O(k \log \log k)$ local search steps obtained through the $k$-means++ sampling distribution to yield a $c$-approximation to the $k$-means clustering problem, where $c$ is a large absolute constant. Here we generalize and extend their local search algorithm by considering larger and more sophisticated local search neighborhoods hence allowing to swap multiple centers at the same time. Our algorithm achieves a $9 + \varepsilon$ approximation ratio, which is the best possible for local search. Importantly we show that our approach yields substantial practical improvements, we show significant quality improvements over the approach of Lattanzi and Sohler 
    

