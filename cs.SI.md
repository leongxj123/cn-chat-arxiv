# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Broadcasting in random recursive dags.](http://arxiv.org/abs/2306.01727) | 该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。 |

# 详细

[^1]: 在随机递归有向无环图中的广播

    Broadcasting in random recursive dags. (arXiv:2306.01727v1 [stat.ML])

    [http://arxiv.org/abs/2306.01727](http://arxiv.org/abs/2306.01727)

    该论文研究了一个均匀的$k$-dag广播模型，确定了与$p$和$k$有关的阈值，并讨论了大多数规则的误差率。

    

    一个均匀的$k$-dag通过从现有节点中均匀随机选择$k$个父节点来推广均匀的随机递归树。它以$k$个“根”开始。每个$k$个根节点都被分配一个位。这些位通过一个嘈杂的信道传播。每个父节点的位都以概率$p$发生变化，并进行大多数表决。当所有节点都接收到它们的位后，$k$-dag被显示，不识别根节点。目标是估计所有根节点中的大多数位。我们确定了$p$的阈值，作为一个关于$k$的函数，使得所有节点的大多数规则产生错误$c+o(1)$的概率小于$1/2$。在阈值以上，大多数规则的错误概率为$1/2+o(1)$。

    A uniform $k$-{\sc dag} generalizes the uniform random recursive tree by picking $k$ parents uniformly at random from the existing nodes. It starts with $k$ ''roots''. Each of the $k$ roots is assigned a bit. These bits are propagated by a noisy channel. The parents' bits are flipped with probability $p$, and a majority vote is taken. When all nodes have received their bits, the $k$-{\sc dag} is shown without identifying the roots. The goal is to estimate the majority bit among the roots. We identify the threshold for $p$ as a function of $k$ below which the majority rule among all nodes yields an error $c+o(1)$ with $c<1/2$. Above the threshold the majority rule errs with probability $1/2+o(1)$.
    

