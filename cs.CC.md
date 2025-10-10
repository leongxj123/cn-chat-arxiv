# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the average-case complexity of learning output distributions of quantum circuits.](http://arxiv.org/abs/2305.05765) | 本文证明了砖墙随机量子电路输出分布学习是一个平均复杂度困难的问题，需要进行超多项式次数的查询才能有效解决。 |
| [^2] | [Strategyproofness-Exposing Mechanism Descriptions.](http://arxiv.org/abs/2209.13148) | 本文研究了菜单描述在展示机制策略无懈可击性方面的作用，提出了一种新的简单菜单描述的延迟接受机制，并通过实验验证了菜单描述的优势和挑战。 |

# 详细

[^1]: 关于量子电路输出分布学习的平均复杂度

    On the average-case complexity of learning output distributions of quantum circuits. (arXiv:2305.05765v1 [quant-ph])

    [http://arxiv.org/abs/2305.05765](http://arxiv.org/abs/2305.05765)

    本文证明了砖墙随机量子电路输出分布学习是一个平均复杂度困难的问题，需要进行超多项式次数的查询才能有效解决。

    

    本文研究了砖墙随机量子电路的输出分布学习问题，并证明在统计查询模型下，该问题的平均复杂度是困难的。具体地，对于深度为$d$、由$n$个量子比特构成的砖墙随机量子电路，我们得出了三个主要结论：在超对数电路深度$d=\omega(\log(n))$时，任何学习算法都需要进行超多项式次数的查询才能在随机实例上实现恒定的成功概率。存在一个$d=O(n)$，这意味着任何学习算法需要进行$\Omega(2^n)$次查询才能在随机实例上实现$O(2^{-n})$的成功概率。在无限电路深度$d\to\infty$时，任何学习算法都需要进行$2^{2^{\Omega(n)}}$次查询才能在随机实例上实现$2^{-2^{\Omega(n)}}$的成功概率。作为一个独立的辅助结果，我们还证明了......（文章内容截断）

    In this work, we show that learning the output distributions of brickwork random quantum circuits is average-case hard in the statistical query model. This learning model is widely used as an abstract computational model for most generic learning algorithms. In particular, for brickwork random quantum circuits on $n$ qubits of depth $d$, we show three main results:  - At super logarithmic circuit depth $d=\omega(\log(n))$, any learning algorithm requires super polynomially many queries to achieve a constant probability of success over the randomly drawn instance.  - There exists a $d=O(n)$, such that any learning algorithm requires $\Omega(2^n)$ queries to achieve a $O(2^{-n})$ probability of success over the randomly drawn instance.  - At infinite circuit depth $d\to\infty$, any learning algorithm requires $2^{2^{\Omega(n)}}$ many queries to achieve a $2^{-2^{\Omega(n)}}$ probability of success over the randomly drawn instance.  As an auxiliary result of independent interest, we show 
    
[^2]: 具有策略无懈可击性的暴露机制描述

    Strategyproofness-Exposing Mechanism Descriptions. (arXiv:2209.13148v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2209.13148](http://arxiv.org/abs/2209.13148)

    本文研究了菜单描述在展示机制策略无懈可击性方面的作用，提出了一种新的简单菜单描述的延迟接受机制，并通过实验验证了菜单描述的优势和挑战。

    

    菜单描述以两个步骤向玩家i展示机制。第一步使用其他玩家的报告描述i的菜单：即i的潜在结果集合。第二步使用i的报告从她的菜单中选择i最喜欢的结果。菜单描述能更好地暴露策略无懈可击性吗，而不会牺牲简单性？我们提出了一个新的简单菜单描述的延迟接受机制。我们证明了，与其他常见的匹配机制相比，这种菜单描述必须与相应的传统描述有着实质性的不同。我们通过对两种基本机制的实验室实验证明了菜单描述的优势和挑战。

    A menu description presents a mechanism to player $i$ in two steps. Step (1) uses the reports of other players to describe $i$'s menu: the set of $i$'s potential outcomes. Step (2) uses $i$'s report to select $i$'s favorite outcome from her menu. Can menu descriptions better expose strategyproofness, without sacrificing simplicity? We propose a new, simple menu description of Deferred Acceptance. We prove that -- in contrast with other common matching mechanisms -- this menu description must differ substantially from the corresponding traditional description. We demonstrate, with a lab experiment on two elementary mechanisms, the promise and challenges of menu descriptions.
    

