# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ant Colony Sampling with GFlowNets for Combinatorial Optimization](https://arxiv.org/abs/2403.07041) | 本文提出了生成流蚁群采样器（GFACS），一种结合生成流网络与蚁群优化方法的神经引导元启发式算法，在组合优化任务中表现优于基线ACO算法并与特定问题启发式方法具有竞争力。 |

# 详细

[^1]: 使用GFlowNets的蚁群采样用于组合优化

    Ant Colony Sampling with GFlowNets for Combinatorial Optimization

    [https://arxiv.org/abs/2403.07041](https://arxiv.org/abs/2403.07041)

    本文提出了生成流蚁群采样器（GFACS），一种结合生成流网络与蚁群优化方法的神经引导元启发式算法，在组合优化任务中表现优于基线ACO算法并与特定问题启发式方法具有竞争力。

    

    本文介绍了生成流蚁群采样器（GFACS），这是一种新颖的用于组合优化的神经引导元启发式算法。GFACS 将生成流网络（GFlowNets）与蚁群优化（ACO）方法相结合。GFlowNets 是一种生成模型，它在组合空间中学习构造性策略，通过在输入图实例上提供决策变量的知情先验分布来增强 ACO。此外，我们引入了一种新颖的训练技巧组合，包括搜索引导的局部探索、能量归一化和能量塑形，以提高 GFACS 的性能。我们的实验结果表明，GFACS 在七个组合优化任务中优于基线 ACO 算法，并且在车辆路径问题的问题特定启发式方法中具有竞争力。源代码可在 \url{https://github.com/ai4co/gfacs} 获取。

    arXiv:2403.07041v1 Announce Type: new  Abstract: This paper introduces the Generative Flow Ant Colony Sampler (GFACS), a novel neural-guided meta-heuristic algorithm for combinatorial optimization. GFACS integrates generative flow networks (GFlowNets) with the ant colony optimization (ACO) methodology. GFlowNets, a generative model that learns a constructive policy in combinatorial spaces, enhance ACO by providing an informed prior distribution of decision variables conditioned on input graph instances. Furthermore, we introduce a novel combination of training tricks, including search-guided local exploration, energy normalization, and energy shaping to improve GFACS. Our experimental results demonstrate that GFACS outperforms baseline ACO algorithms in seven CO tasks and is competitive with problem-specific heuristics for vehicle routing problems. The source code is available at \url{https://github.com/ai4co/gfacs}.
    

