# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Distributed Estimation and Inference for Semi-parametric Binary Response Models](https://arxiv.org/abs/2210.08393) | 通过一次性分治估计和多轮估计，本文提出了分布式环境下对半参数二元选择模型的新估计方法，实现了优化误差的超线性改进。 |

# 详细

[^1]: 分布式估计和推断用于半参数二元响应模型

    Distributed Estimation and Inference for Semi-parametric Binary Response Models

    [https://arxiv.org/abs/2210.08393](https://arxiv.org/abs/2210.08393)

    通过一次性分治估计和多轮估计，本文提出了分布式环境下对半参数二元选择模型的新估计方法，实现了优化误差的超线性改进。

    

    现代技术的发展使得数据收集的规模前所未有，这给许多统计估计和推断问题带来了新挑战。本文研究了在分布式计算环境下对半参数二元选择模型的最大分数估计器，而无需预先指定噪声分布。传统的分治估计器在计算上昂贵，并受到机器数量的非正则约束的限制，这是由于目标函数的高度非光滑性质导致的。我们提出了(1)在平滑目标之后进行一次性分治估计以放宽约束，以及(2)通过迭代平滑完全去除约束的多轮估计。我们指定了一种自适应的核平滑器选择，通过顺序缩小带宽在多次迭代中实现了对优化误差的超线性改进。

    arXiv:2210.08393v3 Announce Type: replace-cross  Abstract: The development of modern technology has enabled data collection of unprecedented size, which poses new challenges to many statistical estimation and inference problems. This paper studies the maximum score estimator of a semi-parametric binary choice model under a distributed computing environment without pre-specifying the noise distribution. An intuitive divide-and-conquer estimator is computationally expensive and restricted by a non-regular constraint on the number of machines, due to the highly non-smooth nature of the objective function. We propose (1) a one-shot divide-and-conquer estimator after smoothing the objective to relax the constraint, and (2) a multi-round estimator to completely remove the constraint via iterative smoothing. We specify an adaptive choice of kernel smoother with a sequentially shrinking bandwidth to achieve the superlinear improvement of the optimization error over the multiple iterations. The
    

