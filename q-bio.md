# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Molecular Generative Adversarial Network with Multi-Property Optimization](https://arxiv.org/abs/2404.00081) | 该研究引入了一种新型的基于演员-评论家强化学习的GAN，即InstGAN，以在令牌级别上生成具有多属性优化的分子，并利用最大化信息熵来缓解模式崩溃。 |
| [^2] | [Diffusion on language model embeddings for protein sequence generation](https://arxiv.org/abs/2403.03726) | 使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。 |
| [^3] | [Toward a Team of AI-made Scientists for Scientific Discovery from Gene Expression Data](https://arxiv.org/abs/2402.12391) | 引入了一个名为AI科学家团队（TAIS）的框架，旨在简化科学发现流程，由模拟角色协作，特别关注于识别具有疾病预测价值的基因 |

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
    
[^3]: 实现基因表达数据科学发现的AI科学家团队

    Toward a Team of AI-made Scientists for Scientific Discovery from Gene Expression Data

    [https://arxiv.org/abs/2402.12391](https://arxiv.org/abs/2402.12391)

    引入了一个名为AI科学家团队（TAIS）的框架，旨在简化科学发现流程，由模拟角色协作，特别关注于识别具有疾病预测价值的基因

    

    机器学习已成为科学发现的强大工具，使研究人员能够从复杂数据集中提取有意义的见解。我们引入了一个新颖的框架，名为AI科学家团队（TAIS），旨在简化科学发现流程。TAIS包括模拟角色，包括项目经理、数据工程师和领域专家，每个角色由大型语言模型（LLM）代表。这些角色协作以复制数据科学家通常执行的任务，特别关注于识别具有疾病预测价值的基因。

    arXiv:2402.12391v1 Announce Type: cross  Abstract: Machine learning has emerged as a powerful tool for scientific discovery, enabling researchers to extract meaningful insights from complex datasets. For instance, it has facilitated the identification of disease-predictive genes from gene expression data, significantly advancing healthcare. However, the traditional process for analyzing such datasets demands substantial human effort and expertise for the data selection, processing, and analysis. To address this challenge, we introduce a novel framework, a Team of AI-made Scientists (TAIS), designed to streamline the scientific discovery pipeline. TAIS comprises simulated roles, including a project manager, data engineer, and domain expert, each represented by a Large Language Model (LLM). These roles collaborate to replicate the tasks typically performed by data scientists, with a specific focus on identifying disease-predictive genes. Furthermore, we have curated a benchmark dataset t
    

