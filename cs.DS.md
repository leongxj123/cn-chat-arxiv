# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the fast convergence of minibatch heavy ball momentum.](http://arxiv.org/abs/2206.07553) | 本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。 |

# 详细

[^1]: 论小批量重球动量法的快速收敛性

    On the fast convergence of minibatch heavy ball momentum. (arXiv:2206.07553v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.07553](http://arxiv.org/abs/2206.07553)

    本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。

    

    简单的随机动量方法被广泛用于机器学习优化中，但由于还没有加速的理论保证，这与它们在实践中的良好性能并不相符。本文旨在通过展示，随机重球动量在二次最优化问题中保持（确定性）重球动量的快速线性率，至少在使用足够大的批量大小进行小批量处理时。我们所研究的算法可以被解释为带小批量处理和重球动量的加速随机Kaczmarz算法。该分析依赖于仔细分解动量转移矩阵，并使用新的独立随机矩阵乘积的谱范围集中界限。我们提供了数值演示，证明了我们的界限相当尖锐。

    Simple stochastic momentum methods are widely used in machine learning optimization, but their good practical performance is at odds with an absence of theoretical guarantees of acceleration in the literature. In this work, we aim to close the gap between theory and practice by showing that stochastic heavy ball momentum retains the fast linear rate of (deterministic) heavy ball momentum on quadratic optimization problems, at least when minibatching with a sufficiently large batch size. The algorithm we study can be interpreted as an accelerated randomized Kaczmarz algorithm with minibatching and heavy ball momentum. The analysis relies on carefully decomposing the momentum transition matrix, and using new spectral norm concentration bounds for products of independent random matrices. We provide numerical illustrations demonstrating that our bounds are reasonably sharp.
    

