# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2402.16359) | 提出了一种反馈高效的在线微调扩散模型的强化学习程序 |
| [^2] | [Revolutionizing Genomics with Reinforcement Learning Techniques.](http://arxiv.org/abs/2302.13268) | 强化学习是一种革新的工具，可以在基因组学领域中解决自动数据分析和处理的问题。使用强化学习算法可以降低收集标记训练数据的成本，适用于基因组数据分析和解释。本调查重点关注在基因组研究领域中使用强化学习的应用，包括基因调控网络、基因组组装和序列比对。 |

# 详细

[^1]: 反馈高效在线微调扩散模型

    Feedback Efficient Online Fine-Tuning of Diffusion Models

    [https://arxiv.org/abs/2402.16359](https://arxiv.org/abs/2402.16359)

    提出了一种反馈高效的在线微调扩散模型的强化学习程序

    

    扩散模型在建模复杂数据分布方面表现出色，包括图像，蛋白质和小分子的分布。然而，在许多情况下，我们的目标是模拟最大化某些属性的分布的部分：例如，我们可能希望生成具有高审美质量的图像，或具有高生物活性的分子。自然地，我们可以将这视为一个强化学习（RL）问题，其目标是微调扩散模型以最大化与某些属性对应的奖励函数。即使可以访问地面真实奖励函数的在线查询，有效地发现高奖励样本也可能具有挑战性：它们在初始分布中的概率可能很低，并且可能存在许多不可行的样本，甚至没有定义良好的奖励（例如，不自然的图像或物理上不可能的分子）。在这项工作中，我们提出了一种新颖的强化学习程序，可以高效地发现高奖励样本。

    arXiv:2402.16359v1 Announce Type: cross  Abstract: Diffusion models excel at modeling complex data distributions, including those of images, proteins, and small molecules. However, in many cases, our goal is to model parts of the distribution that maximize certain properties: for example, we may want to generate images with high aesthetic quality, or molecules with high bioactivity. It is natural to frame this as a reinforcement learning (RL) problem, in which the objective is to fine-tune a diffusion model to maximize a reward function that corresponds to some property. Even with access to online queries of the ground-truth reward function, efficiently discovering high-reward samples can be challenging: they might have a low probability in the initial distribution, and there might be many infeasible samples that do not even have a well-defined reward (e.g., unnatural images or physically impossible molecules). In this work, we propose a novel reinforcement learning procedure that effi
    
[^2]: 使用强化学习技术革新基因组学

    Revolutionizing Genomics with Reinforcement Learning Techniques. (arXiv:2302.13268v2 [q-bio.GN] UPDATED)

    [http://arxiv.org/abs/2302.13268](http://arxiv.org/abs/2302.13268)

    强化学习是一种革新的工具，可以在基因组学领域中解决自动数据分析和处理的问题。使用强化学习算法可以降低收集标记训练数据的成本，适用于基因组数据分析和解释。本调查重点关注在基因组研究领域中使用强化学习的应用，包括基因调控网络、基因组组装和序列比对。

    

    近年来，强化学习（RL）作为一种强大的工具出现在解决各种问题中，包括决策和基因组学。过去二十年的原始基因组数据指数增长已经超出了手动分析的能力，这导致对自动数据分析和处理的兴趣越来越大。RL算法能够在最小的人工监督下从经验中学习，使其非常适合基因组数据分析和解释。使用RL的一个关键好处是降低了收集标记训练数据的成本，这是监督学习所需的。虽然已经有许多研究探讨了机器学习在基因组学中的应用，但本调查仅专注于在各种基因组研究领域（包括基因调控网络，基因组组装和序列比对）中使用RL的情况。我们对现有研究的技术细节进行了全面的概述。

    In recent years, Reinforcement Learning (RL) has emerged as a powerful tool for solving a wide range of problems, including decision-making and genomics. The exponential growth of raw genomic data over the past two decades has exceeded the capacity of manual analysis, leading to a growing interest in automatic data analysis and processing. RL algorithms are capable of learning from experience with minimal human supervision, making them well-suited for genomic data analysis and interpretation. One of the key benefits of using RL is the reduced cost associated with collecting labeled training data, which is required for supervised learning. While there have been numerous studies examining the applications of Machine Learning (ML) in genomics, this survey focuses exclusively on the use of RL in various genomics research fields, including gene regulatory networks (GRNs), genome assembly, and sequence alignment. We present a comprehensive technical overview of existing studies on the applic
    

