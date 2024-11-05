# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Query-Efficient Correlation Clustering with Noisy Oracle](https://rss.arxiv.org/abs/2402.01400) | 本论文提出了一种低查询成本的聚类方法，利用纯在组合多臂赌博机探索范式实现在线学习，并设计了能在NP-hard情况下运行的多项式时间算法。 |
| [^2] | [Optimization on a Finer Scale: Bounded Local Subgradient Variation Perspective](https://arxiv.org/abs/2403.16317) | 该研究在有界局部次梯度变化的条件下研究非光滑优化问题，提出的目标函数类能帮助更好理解传统优化问题的复杂性，并在一般情况下降低oracle复杂度。 |

# 详细

[^1]: 低查询成本带噪声or同时聚类

    Query-Efficient Correlation Clustering with Noisy Oracle

    [https://rss.arxiv.org/abs/2402.01400](https://rss.arxiv.org/abs/2402.01400)

    本论文提出了一种低查询成本的聚类方法，利用纯在组合多臂赌博机探索范式实现在线学习，并设计了能在NP-hard情况下运行的多项式时间算法。

    

    我们研究了一个常见的聚类设置，其中我们需要对n个元素进行聚类，并且我们的目标是尽可能少地向返回两个元素相似性的有噪声的oracle查询。我们的设置涵盖了许多应用领域，在这些领域中，相似性函数计算起来成本高并且 inherently noisy。我们提出了两种基于纯在组合多臂赌博机探索范式(PE-CMAB)的在线学习问题的新颖表达方法固定置信度和固定预算设置。对于这两种设置，我们设计了将抽样策略与经典的相关聚类近似算法相结合的算法，并研究了它们的理论保证。我们的结果是这样的：这些算法是第一个在底层离线优化问题为NP-hard的情况下运行的多项式时间算法的例子。

    We study a general clustering setting in which we have $n$ elements to be clustered, and we aim to perform as few queries as possible to an oracle that returns a noisy sample of the similarity between two elements. Our setting encompasses many application domains in which the similarity function is costly to compute and inherently noisy. We propose two novel formulations of online learning problems rooted in the paradigm of Pure Exploration in Combinatorial Multi-Armed Bandits (PE-CMAB): fixed confidence and fixed budget settings. For both settings, we design algorithms that combine a sampling strategy with a classic approximation algorithm for correlation clustering and study their theoretical guarantees. Our results are the first examples of polynomial-time algorithms that work for the case of PE-CMAB in which the underlying offline optimization problem is NP-hard.
    
[^2]: 在更细粒度上的优化：有界局部次梯度变化的视角

    Optimization on a Finer Scale: Bounded Local Subgradient Variation Perspective

    [https://arxiv.org/abs/2403.16317](https://arxiv.org/abs/2403.16317)

    该研究在有界局部次梯度变化的条件下研究非光滑优化问题，提出的目标函数类能帮助更好理解传统优化问题的复杂性，并在一般情况下降低oracle复杂度。

    

    我们在有界局部次梯度变化的条件下开始研究非光滑优化问题，它假设在点附近的小区域内，(次)梯度之间存在有限的差异，可以用平均或最大方式求值。由此产生的目标函数类包括传统优化中传统研究的目标函数类，这些类根据目标函数的Lipschitz连续性或其梯度的H\"{o}lder/Lipschitz连续性定义。此外，该定义类包含那些既不是Lipschitz连续的也没有H\"{o}lder连续梯度的函数。当限制在传统优化问题类时，定义研究类的参数导致更加精细的复杂性界限，在最坏情况下恢复传统的oracle复杂度界限，但一般情况下会导致那些不是“最坏情况”的函数具有较低的oracle 复杂性。

    arXiv:2403.16317v1 Announce Type: cross  Abstract: We initiate the study of nonsmooth optimization problems under bounded local subgradient variation, which postulates bounded difference between (sub)gradients in small local regions around points, in either average or maximum sense. The resulting class of objective functions encapsulates the classes of objective functions traditionally studied in optimization, which are defined based on either Lipschitz continuity of the objective or H\"{o}lder/Lipschitz continuity of its gradient. Further, the defined class contains functions that are neither Lipschitz continuous nor have a H\"{o}lder continuous gradient. When restricted to the traditional classes of optimization problems, the parameters defining the studied classes lead to more fine-grained complexity bounds, recovering traditional oracle complexity bounds in the worst case but generally leading to lower oracle complexity for functions that are not ``worst case.'' Some highlights of 
    

