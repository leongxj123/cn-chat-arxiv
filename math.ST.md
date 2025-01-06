# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Parallelized Midpoint Randomization for Langevin Monte Carlo](https://arxiv.org/abs/2402.14434) | 探索在能够进行梯度平行评估的框架中的抽样问题，提出了并行化的随机中点方法，并通过新技术导出了对抽样和目标密度之间Wasserstein距离的上界，量化了并行处理单元带来的运行时改进。 |

# 详细

[^1]: 并行中点随机化的 Langevin Monte Carlo

    Parallelized Midpoint Randomization for Langevin Monte Carlo

    [https://arxiv.org/abs/2402.14434](https://arxiv.org/abs/2402.14434)

    探索在能够进行梯度平行评估的框架中的抽样问题，提出了并行化的随机中点方法，并通过新技术导出了对抽样和目标密度之间Wasserstein距离的上界，量化了并行处理单元带来的运行时改进。

    

    我们探讨了在可以进行梯度的平行评估的框架中的抽样问题。我们的研究重点放在由平滑和强log-凹密度表征的目标分布上。我们重新审视了并行化的随机中点方法，并运用最近开发用于分析其纯顺序版本的证明技术。利用这些技术，我们得出了抽样和目标密度之间的Wasserstein距离的上界。这些界限量化了通过利用并行处理单元所实现的运行时改进，这可能是相当可观的。

    arXiv:2402.14434v1 Announce Type: cross  Abstract: We explore the sampling problem within the framework where parallel evaluations of the gradient of the log-density are feasible. Our investigation focuses on target distributions characterized by smooth and strongly log-concave densities. We revisit the parallelized randomized midpoint method and employ proof techniques recently developed for analyzing its purely sequential version. Leveraging these techniques, we derive upper bounds on the Wasserstein distance between the sampling and target densities. These bounds quantify the runtime improvement achieved by utilizing parallel processing units, which can be considerable.
    

