# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved Hardness Results for Learning Intersections of Halfspaces](https://arxiv.org/abs/2402.15995) | 我们通过展示学习在维度N中的$\omega(\log \log N)$个半空间甚至需要超多项式时间的标准假设，显著缩小了这一差距 |
| [^2] | [Consistency of Lloyd's Algorithm Under Perturbations.](http://arxiv.org/abs/2309.00578) | 该论文研究了Lloyd算法在扰动样本上的一致性，证明了在适当初始化和扰动相对于亚高斯噪声较小的假设下，算法在O(log(n))次迭代后的错聚类率在指数下界受限。 |

# 详细

[^1]: 改进学习半空间交集的困难性结果

    Improved Hardness Results for Learning Intersections of Halfspaces

    [https://arxiv.org/abs/2402.15995](https://arxiv.org/abs/2402.15995)

    我们通过展示学习在维度N中的$\omega(\log \log N)$个半空间甚至需要超多项式时间的标准假设，显著缩小了这一差距

    

    我们展示了在不正确设置中学习半空间交集的弱学习下界，这些下界非常强大（并且令人惊讶地简单）。关于这个问题知之甚少。例如，甚至不知道是否存在一个多项式时间算法来学习仅两个半空间的交集。另一方面，基于良好建立的假设（如近似最坏情况的格问题或Feige的3SAT假设的变体）的下界仅对超对数个半空间的交集已知（或者由已有结果暗示）。

    arXiv:2402.15995v1 Announce Type: cross  Abstract: We show strong (and surprisingly simple) lower bounds for weakly learning intersections of halfspaces in the improper setting. Strikingly little is known about this problem. For instance, it is not even known if there is a polynomial-time algorithm for learning the intersection of only two halfspaces. On the other hand, lower bounds based on well-established assumptions (such as approximating worst-case lattice problems or variants of Feige's 3SAT hypothesis) are only known (or are implied by existing results) for the intersection of super-logarithmically many halfspaces [KS09,KS06,DSS16]. With intersections of fewer halfspaces being only ruled out under less standard assumptions [DV21] (such as the existence of local pseudo-random generators with large stretch). We significantly narrow this gap by showing that even learning $\omega(\log \log N)$ halfspaces in dimension $N$ takes super-polynomial time under standard assumptions on wors
    
[^2]: Lloyd算法在扰动下的一致性

    Consistency of Lloyd's Algorithm Under Perturbations. (arXiv:2309.00578v1 [cs.LG])

    [http://arxiv.org/abs/2309.00578](http://arxiv.org/abs/2309.00578)

    该论文研究了Lloyd算法在扰动样本上的一致性，证明了在适当初始化和扰动相对于亚高斯噪声较小的假设下，算法在O(log(n))次迭代后的错聚类率在指数下界受限。

    

    在无监督学习的背景下，Lloyd算法是最常用的聚类算法之一。它启发了大量的工作，研究了算法在不同设置下对地面真实聚类的正确性。特别是在2016年，卢和周表明，在正确初始化算法的前提下，Lloyd算法在从亚高斯混合中独立抽取的n个样本上的错聚类率在O(log(n))次迭代后指数下界受限。然而，在许多应用中，真实样本是未观测到的，需要通过预处理流水线（如合适的数据矩阵上的谱方法）从数据中学习。我们展示了在适当初始化和扰动相对于亚高斯噪声较小的假设下，Lloyd算法在从亚高斯混合中扰动样本上的错聚类率在O(log(n))次迭代后同样指数下界受限。

    In the context of unsupervised learning, Lloyd's algorithm is one of the most widely used clustering algorithms. It has inspired a plethora of work investigating the correctness of the algorithm under various settings with ground truth clusters. In particular, in 2016, Lu and Zhou have shown that the mis-clustering rate of Lloyd's algorithm on $n$ independent samples from a sub-Gaussian mixture is exponentially bounded after $O(\log(n))$ iterations, assuming proper initialization of the algorithm. However, in many applications, the true samples are unobserved and need to be learned from the data via pre-processing pipelines such as spectral methods on appropriate data matrices. We show that the mis-clustering rate of Lloyd's algorithm on perturbed samples from a sub-Gaussian mixture is also exponentially bounded after $O(\log(n))$ iterations under the assumptions of proper initialization and that the perturbation is small relative to the sub-Gaussian noise. In canonical settings with g
    

