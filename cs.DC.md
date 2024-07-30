# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation](https://arxiv.org/abs/2311.09922) | 本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。 |
| [^2] | [Robust Fully-Asynchronous Methods for Distributed Training over General Architecture.](http://arxiv.org/abs/2307.11617) | 该论文提出了一种强健的全异步方法（R-FAST），在分布式机器学习中解决了完美同步的低效性和不可能性问题，通过采用鲁棒的梯度跟踪策略和灵活的通信架构，消除了数据异构性和数据包丢失的影响，并实现了期望的邻域收敛。 |

# 详细

[^1]: 通过采用整数列表作为多项式基数2指数的集合来实现快速乘法

    Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation

    [https://arxiv.org/abs/2311.09922](https://arxiv.org/abs/2311.09922)

    本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。

    

    我们演示了一种基于用整数列表表示的多项式基数2指数集合的乘法方法。该方法采用python代码实现了一组算法。我们展示了该方法在某一位数范围内比数论变换(NTT)和卡拉茨巴(Karatsuba)乘法更快。我们还实现了用python代码进行比较，与多项式基数2整数方法进行比较。我们展示了任何整数或实数都可以表示为整数索引列表，表示二进制中的有限级数。该数字的整数索引有限级数可以存储和分布在多个CPU / GPU上。我们展示了加法和乘法运算可以应用于作为索引整数表示的两个补码加法，并可以完全分布在给定的CPU / GPU架构上。我们展示了完全的分布性能。

    We demonstrate a multiplication method based on numbers represented as set of polynomial radix 2 indices stored as an integer list. The 'polynomial integer index multiplication' method is a set of algorithms implemented in python code. We demonstrate the method to be faster than both the Number Theoretic Transform (NTT) and Karatsuba for multiplication within a certain bit range. Also implemented in python code for comparison purposes with the polynomial radix 2 integer method. We demonstrate that it is possible to express any integer or real number as a list of integer indices, representing a finite series in base two. The finite series of integer index representation of a number can then be stored and distributed across multiple CPUs / GPUs. We show that operations of addition and multiplication can be applied as two's complement additions operating on the index integer representations and can be fully distributed across a given CPU / GPU architecture. We demonstrate fully distribute
    
[^2]: 强健的全异步方法用于通用架构上的分布式训练

    Robust Fully-Asynchronous Methods for Distributed Training over General Architecture. (arXiv:2307.11617v1 [cs.DC])

    [http://arxiv.org/abs/2307.11617](http://arxiv.org/abs/2307.11617)

    该论文提出了一种强健的全异步方法（R-FAST），在分布式机器学习中解决了完美同步的低效性和不可能性问题，通过采用鲁棒的梯度跟踪策略和灵活的通信架构，消除了数据异构性和数据包丢失的影响，并实现了期望的邻域收敛。

    

    分布式机器学习问题中完美的同步是低效甚至不可能的，由于延迟、数据丢失和延迟较大的设备。我们提出了一种强健的全异步随机梯度跟踪方法（R-FAST），其中每个设备以自己的速度进行本地计算和通信，而无需任何形式的同步。与现有的异步分布式算法不同，R-FAST可以通过采用基于设计良好的辅助变量来跟踪和缓冲整体梯度向量的鲁棒梯度跟踪策略，消除设备间数据异构性的影响，并允许数据包丢失。更重要的是，所提出的方法利用两个生成树图进行通信，只要两者共享至少一个共同的根节点，就能实现灵活的通信架构设计。我们证明了R-FAST对于平滑和强凸问题，收敛到最优解的期望邻域，并具有几何收敛率。

    Perfect synchronization in distributed machine learning problems is inefficient and even impossible due to the existence of latency, package losses and stragglers. We propose a Robust Fully-Asynchronous Stochastic Gradient Tracking method (R-FAST), where each device performs local computation and communication at its own pace without any form of synchronization. Different from existing asynchronous distributed algorithms, R-FAST can eliminate the impact of data heterogeneity across devices and allow for packet losses by employing a robust gradient tracking strategy that relies on properly designed auxiliary variables for tracking and buffering the overall gradient vector. More importantly, the proposed method utilizes two spanning-tree graphs for communication so long as both share at least one common root, enabling flexible designs in communication architectures. We show that R-FAST converges in expectation to a neighborhood of the optimum with a geometric rate for smooth and strongly
    

