# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lower Bounds and Accelerated Algorithms in Distributed Stochastic Optimization with Communication Compression.](http://arxiv.org/abs/2305.07612) | 本文研究采用通信压缩的分布式随机优化算法的性能下限并提出了一种名为NEOLITHIC的新型通信压缩算法，通过加速收敛速率缩小下限和现有算法的差距。 |

# 详细

[^1]: 通信压缩下的分布式随机优化中的下限和加速算法

    Lower Bounds and Accelerated Algorithms in Distributed Stochastic Optimization with Communication Compression. (arXiv:2305.07612v1 [cs.LG])

    [http://arxiv.org/abs/2305.07612](http://arxiv.org/abs/2305.07612)

    本文研究采用通信压缩的分布式随机优化算法的性能下限并提出了一种名为NEOLITHIC的新型通信压缩算法，通过加速收敛速率缩小下限和现有算法的差距。

    

    通信压缩是减轻分布式随机优化中计算节点间信息交换量的重要策略。本文研究了采用通信压缩的分布式随机优化算法的性能下限，并关注两种主要类型的压缩器：无偏和压缩型，并解决了可以通过这些压缩器获得的最佳收敛速率问题。本文针对六种不同设置，结合强凸、一般凸或非凸函数，并用无偏或压缩型压缩器建立了分布式随机优化的收敛速率下限。我们提出了一种名为NEOLITHIC的新型通信压缩算法，通过加速收敛速率相比经典方法，缩小了下限和现有算法的差距。理论分析和数值实验提供了关于分布式随机优化中通信压缩算法的最优性能的见解。

    Communication compression is an essential strategy for alleviating communication overhead by reducing the volume of information exchanged between computing nodes in large-scale distributed stochastic optimization. Although numerous algorithms with convergence guarantees have been obtained, the optimal performance limit under communication compression remains unclear.  In this paper, we investigate the performance limit of distributed stochastic optimization algorithms employing communication compression. We focus on two main types of compressors, unbiased and contractive, and address the best-possible convergence rates one can obtain with these compressors. We establish the lower bounds for the convergence rates of distributed stochastic optimization in six different settings, combining strongly-convex, generally-convex, or non-convex functions with unbiased or contractive compressor types. To bridge the gap between lower bounds and existing algorithms' rates, we propose NEOLITHIC, a n
    

