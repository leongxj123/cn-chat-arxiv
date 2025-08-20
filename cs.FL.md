# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Learning of Mealy Machines with Timers](https://arxiv.org/abs/2403.02019) | 这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。 |

# 详细

[^1]: 具有定时器的Mealy机器的主动学习

    Active Learning of Mealy Machines with Timers

    [https://arxiv.org/abs/2403.02019](https://arxiv.org/abs/2403.02019)

    这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。

    

    我们在黑盒环境中提出了第一个用于查询学习一般类别的具有定时器的Mealy机器（MMTs）的算法。我们的算法是Vaandrager等人的L＃算法对定时设置的扩展。类似于Waga提出的用于学习定时自动机的算法，我们的算法受到Maler＆Pnueli思想的启发。我们的算法和Waga的算法都使用符号查询进行基础语言学习，然后使用有限数量的具体查询进行实现。然而，Waga需要指数级的具体查询来实现单个符号查询，而我们只需要多项式数量。这是因为要学习定时自动机，学习者需要确定每个转换的确切卫兵和重置（有指数多种可能性），而要学习MMT，学习者只需要弄清楚哪些先前的转换导致超时。正如我们之前的工作所示，

    arXiv:2403.02019v1 Announce Type: cross  Abstract: We present the first algorithm for query learning of a general class of Mealy machines with timers (MMTs) in a black-box context. Our algorithm is an extension of the L# algorithm of Vaandrager et al. to a timed setting. Like the algorithm for learning timed automata proposed by Waga, our algorithm is inspired by ideas of Maler & Pnueli. Based on the elementary languages of, both Waga's and our algorithm use symbolic queries, which are then implemented using finitely many concrete queries. However, whereas Waga needs exponentially many concrete queries to implement a single symbolic query, we only need a polynomial number. This is because in order to learn a timed automaton, a learner needs to determine the exact guard and reset for each transition (out of exponentially many possibilities), whereas for learning an MMT a learner only needs to figure out which of the preceding transitions caused a timeout. As shown in our previous work, 
    

