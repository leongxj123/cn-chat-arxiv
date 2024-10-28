# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Diffusion Policy Optimization](https://arxiv.org/abs/2402.16302) | 本文引入了图扩散策略优化（GDPO），通过强化学习为任意目标优化图扩散模型，实现了在各种图生成任务中的最先进性能。 |

# 详细

[^1]: 图扩散策略优化

    Graph Diffusion Policy Optimization

    [https://arxiv.org/abs/2402.16302](https://arxiv.org/abs/2402.16302)

    本文引入了图扩散策略优化（GDPO），通过强化学习为任意目标优化图扩散模型，实现了在各种图生成任务中的最先进性能。

    

    最近的研究在优化扩散模型以实现特定下游目标方面取得了重要进展，这对于领域如药物设计中的图生成是一个重要的追求。然而，直接将这些模型应用于图扩散存在挑战，导致性能不佳。本文介绍了一种名为图扩散策略优化（GDPO）的新方法，该方法通过强化学习为任意（如非可微分）目标优化图扩散模型。GDPO基于针对图扩散模型量身定制的急切策略梯度，通过认真分析开发，有望提高性能。实验结果表明，GDPO在具有复杂和多样化目标的各种图生成任务中实现了最先进的性能。代码可在https://github.com/sail-sg/GDPO上找到。

    arXiv:2402.16302v1 Announce Type: cross  Abstract: Recent research has made significant progress in optimizing diffusion models for specific downstream objectives, which is an important pursuit in fields such as graph generation for drug design. However, directly applying these models to graph diffusion presents challenges, resulting in suboptimal performance. This paper introduces graph diffusion policy optimization (GDPO), a novel approach to optimize graph diffusion models for arbitrary (e.g., non-differentiable) objectives using reinforcement learning. GDPO is based on an eager policy gradient tailored for graph diffusion models, developed through meticulous analysis and promising improved performance. Experimental results show that GDPO achieves state-of-the-art performance in various graph generation tasks with complex and diverse objectives. Code is available at https://github.com/sail-sg/GDPO.
    

