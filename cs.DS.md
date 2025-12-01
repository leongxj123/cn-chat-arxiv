# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation](https://arxiv.org/abs/2311.09922) | 本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。 |
| [^2] | [Receiver-Oriented Cheap Talk Design.](http://arxiv.org/abs/2401.03671) | 本文研究了接收方导向的廉价谈话设计，提出了透明动机和过滤信息两种模型，并证明在透明动机下接收方不会从过滤信息中获益。然而，一般情况下接收方可以通过过滤获得严格的好处，并提供了计算最优均衡的有效算法。这对于用户控制信息的平台具有创新性，揭示了通信动态并提供了策略性互动预测。 |

# 详细

[^1]: 通过采用整数列表作为多项式基数2指数的集合来实现快速乘法

    Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation

    [https://arxiv.org/abs/2311.09922](https://arxiv.org/abs/2311.09922)

    本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。

    

    我们演示了一种基于用整数列表表示的多项式基数2指数集合的乘法方法。该方法采用python代码实现了一组算法。我们展示了该方法在某一位数范围内比数论变换(NTT)和卡拉茨巴(Karatsuba)乘法更快。我们还实现了用python代码进行比较，与多项式基数2整数方法进行比较。我们展示了任何整数或实数都可以表示为整数索引列表，表示二进制中的有限级数。该数字的整数索引有限级数可以存储和分布在多个CPU / GPU上。我们展示了加法和乘法运算可以应用于作为索引整数表示的两个补码加法，并可以完全分布在给定的CPU / GPU架构上。我们展示了完全的分布性能。

    We demonstrate a multiplication method based on numbers represented as set of polynomial radix 2 indices stored as an integer list. The 'polynomial integer index multiplication' method is a set of algorithms implemented in python code. We demonstrate the method to be faster than both the Number Theoretic Transform (NTT) and Karatsuba for multiplication within a certain bit range. Also implemented in python code for comparison purposes with the polynomial radix 2 integer method. We demonstrate that it is possible to express any integer or real number as a list of integer indices, representing a finite series in base two. The finite series of integer index representation of a number can then be stored and distributed across multiple CPUs / GPUs. We show that operations of addition and multiplication can be applied as two's complement additions operating on the index integer representations and can be fully distributed across a given CPU / GPU architecture. We demonstrate fully distribute
    
[^2]: 接收方导向的廉价谈话设计

    Receiver-Oriented Cheap Talk Design. (arXiv:2401.03671v1 [cs.GT])

    [http://arxiv.org/abs/2401.03671](http://arxiv.org/abs/2401.03671)

    本文研究了接收方导向的廉价谈话设计，提出了透明动机和过滤信息两种模型，并证明在透明动机下接收方不会从过滤信息中获益。然而，一般情况下接收方可以通过过滤获得严格的好处，并提供了计算最优均衡的有效算法。这对于用户控制信息的平台具有创新性，揭示了通信动态并提供了策略性互动预测。

    

    本文考虑了发送方和接收方之间廉价谈话交互的动态，与传统模型的不同之处在于侧重于接收方的角度。我们研究了两种模型，一种有透明的动机，另一种是接收方可以\emph{过滤}发送方可以访问的信息。我们给出了在透明动机下最佳接收方均衡的几何特征，并证明接收方在这种情况下不会从过滤信息中获益。然而，一般情况下，我们显示接收方可以通过过滤获得严格的好处，并提供计算最优均衡的有效算法。这种创新性的分析与用户导向的平台一致，在这些平台上，接收方（用户）控制发送方（卖家）可以访问的信息。我们的发现揭示了通信动态，弥平了发送方固有的优势，并提供了策略性互动预测。

    This paper considers the dynamics of cheap talk interactions between a sender and receiver, departing from conventional models by focusing on the receiver's perspective. We study two models, one with transparent motives and another one in which the receiver can \emph{filter} the information that is accessible by the sender. We give a geometric characterization of the best receiver equilibrium under transparent motives and prove that the receiver does not benefit from filtering information in this case. However, in general, we show that the receiver can strictly benefit from filtering and provide efficient algorithms for computing optimal equilibria. This innovative analysis aligns with user-based platforms where receivers (users) control information accessible to senders (sellers). Our findings provide insights into communication dynamics, leveling the sender's inherent advantage, and offering strategic interaction predictions.
    

