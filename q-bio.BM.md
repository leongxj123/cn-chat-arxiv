# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Molecular Generative Adversarial Network with Multi-Property Optimization](https://arxiv.org/abs/2404.00081) | 该研究引入了一种新型的基于演员-评论家强化学习的GAN，即InstGAN，以在令牌级别上生成具有多属性优化的分子，并利用最大化信息熵来缓解模式崩溃。 |
| [^2] | [Diffusion on language model embeddings for protein sequence generation](https://arxiv.org/abs/2403.03726) | 使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。 |

# 详细

[^1]: 具有多属性优化的分子生成对抗网络

    Molecular Generative Adversarial Network with Multi-Property Optimization

    [https://arxiv.org/abs/2404.00081](https://arxiv.org/abs/2404.00081)

    该研究引入了一种新型的基于演员-评论家强化学习的GAN，即InstGAN，以在令牌级别上生成具有多属性优化的分子，并利用最大化信息熵来缓解模式崩溃。

    

    深度生成模型，如生成对抗网络（GANs），已被应用于药物发现中$de~novo$分子生成。大多数先前的研究使用强化学习（RL）算法，特别是蒙特卡罗树搜索（MCTS），来处理GANs中分子表示的离散特性。然而，由于GANs和RL模型的固有训练不稳定性，以及与MCTS采样相关的高计算成本，MCTS RL-based GANs难以扩展到大型化学数据库。为了解决这些挑战，本研究提出了一种基于带即时和全局奖励的演员-评论家RL的新型GAN，称为InstGAN，以在令牌级别上生成具有多属性优化的分子。此外，最大化信息熵被利用来缓解模式崩溃。实验结果表明，InstGAN优于其他基线，达到了可比较的性能。

    arXiv:2404.00081v1 Announce Type: cross  Abstract: Deep generative models, such as generative adversarial networks (GANs), have been employed for $de~novo$ molecular generation in drug discovery. Most prior studies have utilized reinforcement learning (RL) algorithms, particularly Monte Carlo tree search (MCTS), to handle the discrete nature of molecular representations in GANs. However, due to the inherent instability in training GANs and RL models, along with the high computational cost associated with MCTS sampling, MCTS RL-based GANs struggle to scale to large chemical databases. To tackle these challenges, this study introduces a novel GAN based on actor-critic RL with instant and global rewards, called InstGAN, to generate molecules at the token-level with multi-property optimization. Furthermore, maximized information entropy is leveraged to alleviate the mode collapse. The experimental results demonstrate that InstGAN outperforms other baselines, achieves comparable performance
    
[^2]: 蛋白质序列生成的语言模型嵌入扩散

    Diffusion on language model embeddings for protein sequence generation

    [https://arxiv.org/abs/2403.03726](https://arxiv.org/abs/2403.03726)

    使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。

    

    蛋白设计需要对蛋白质宇宙固有复杂性的深入了解。尽管许多工作倾向于有条件的生成或专注于特定蛋白质家族，但无条件生成的基础任务仍未得到充分探索和重视。在这里，我们探索这个关键领域，引入了DiMA，这是一个利用从蛋白语言模型ESM-2衍生的嵌入进行连续扩散以生成氨基酸序列的模型。DiMA超越了包括自回归变换器和离散扩散模型在内的主要解决方案，我们定量地说明了导致其卓越性能的设计选择所带来的影响。我们使用各种指标跨多种形式广泛评估生成序列的质量、多样性、分布相似性和生物相关性。我们的方法始终产生新颖、多样化的蛋白质序列，精准

    arXiv:2403.03726v1 Announce Type: cross  Abstract: Protein design requires a deep understanding of the inherent complexities of the protein universe. While many efforts lean towards conditional generation or focus on specific families of proteins, the foundational task of unconditional generation remains underexplored and undervalued. Here, we explore this pivotal domain, introducing DiMA, a model that leverages continuous diffusion on embeddings derived from the protein language model, ESM-2, to generate amino acid sequences. DiMA surpasses leading solutions, including autoregressive transformer-based and discrete diffusion models, and we quantitatively illustrate the impact of the design choices that lead to its superior performance. We extensively evaluate the quality, diversity, distribution similarity, and biological relevance of the generated sequences using multiple metrics across various modalities. Our approach consistently produces novel, diverse protein sequences that accura
    

