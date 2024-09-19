# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Graph Generator: Feature-Conditioned Graph Generation using Latent Diffusion Models](https://arxiv.org/abs/2403.01535) | NGG是一个神经图生成器，通过潜在扩散模型实现了特征条件图生成，具有模拟复杂图模式和控制图生成过程的显著能力。 |
| [^2] | [LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education](https://arxiv.org/abs/2402.06264) | 本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。 |
| [^3] | [The Bayan Algorithm: Detecting Communities in Networks Through Exact and Approximate Optimization of Modularity.](http://arxiv.org/abs/2209.04562) | 提出了一种名为Bayan的社区检测算法，通过精确或近似优化模块度的方法，它能够返回最优或接近最优的分区，并且比其他算法快数倍，并能够在合成和真实网络数据集上准确地找到地面真实社区。 |

# 详细

[^1]: 神经图生成器：使用潜在扩散模型的特征条件图生成

    Neural Graph Generator: Feature-Conditioned Graph Generation using Latent Diffusion Models

    [https://arxiv.org/abs/2403.01535](https://arxiv.org/abs/2403.01535)

    NGG是一个神经图生成器，通过潜在扩散模型实现了特征条件图生成，具有模拟复杂图模式和控制图生成过程的显著能力。

    

    图生成已经成为机器学习中的一个关键任务，面临着生成能够准确反映特定属性的图形的重大挑战。本文介绍了神经图生成器（NGG）这一新颖方法，它利用条件潜在扩散模型进行图生成。NGG展示了对复杂图模式进行建模的显著能力，提供了对图生成过程的控制。NGG利用变分图自动编码器进行图压缩，利用在潜在向量空间中的扩散过程，由总结图统计信息的向量指导。我们展示了NGG在各种图生成任务中的通用性，展示了其捕捉期望图属性并推广到未见图形的能力。

    arXiv:2403.01535v1 Announce Type: new  Abstract: Graph generation has emerged as a crucial task in machine learning, with significant challenges in generating graphs that accurately reflect specific properties. Existing methods often fall short in efficiently addressing this need as they struggle with the high-dimensional complexity and varied nature of graph properties. In this paper, we introduce the Neural Graph Generator (NGG), a novel approach which utilizes conditioned latent diffusion models for graph generation. NGG demonstrates a remarkable capacity to model complex graph patterns, offering control over the graph generation process. NGG employs a variational graph autoencoder for graph compression and a diffusion process in the latent vector space, guided by vectors summarizing graph statistics. We demonstrate NGG's versatility across various graph generation tasks, showing its capability to capture desired graph properties and generalize to unseen graphs. This work signifies 
    
[^2]: LLaVA-Docent：利用多模态大型语言模型支持艺术鉴赏教育的教学调优

    LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education

    [https://arxiv.org/abs/2402.06264](https://arxiv.org/abs/2402.06264)

    本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。

    

    艺术鉴赏对于培养学习者的批判性思维和情感智力至关重要。然而，传统的艺术鉴赏教育常面临艺术资源有限的问题，特别是对于弱势学生，并且在主流教育中过度强调科学技术工程和数学科目。为了应对这些挑战，最近的技术进步为创新解决方案铺平了道路。本研究探索了多模态大型语言模型（MLLM）在艺术鉴赏教育中的应用，重点是开发了LLaVA-Docent模型来利用这些进展。我们的方法包括全面的文献综述和与领域专家的咨询，从而形成了一个强大的数据框架。利用这个框架，我们生成了一个虚拟对话数据集，该数据集被GPT-4利用。这个数据集对于训练MLLM（即LLaVA-Docent）起到了关键作用。六名研究人员进行了定量和定性评估。

    Art appreciation is vital in nurturing critical thinking and emotional intelligence among learners. However, traditional art appreciation education has often been hindered by limited access to art resources, especially for disadvantaged students, and an imbalanced emphasis on STEM subjects in mainstream education. In response to these challenges, recent technological advancements have paved the way for innovative solutions. This study explores the application of multi-modal large language models (MLLMs) in art appreciation education, focusing on developing LLaVA-Docent, a model that leverages these advancements. Our approach involved a comprehensive literature review and consultations with experts in the field, leading to developing a robust data framework. Utilizing this framework, we generated a virtual dialogue dataset that was leveraged by GPT-4. This dataset was instrumental in training the MLLM, named LLaVA-Docent. Six researchers conducted quantitative and qualitative evaluation
    
[^3]: Bayan算法：通过对模块度的精确和近似优化来检测网络中的社区

    The Bayan Algorithm: Detecting Communities in Networks Through Exact and Approximate Optimization of Modularity. (arXiv:2209.04562v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2209.04562](http://arxiv.org/abs/2209.04562)

    提出了一种名为Bayan的社区检测算法，通过精确或近似优化模块度的方法，它能够返回最优或接近最优的分区，并且比其他算法快数倍，并能够在合成和真实网络数据集上准确地找到地面真实社区。

    

    社区检测是网络科学中的经典问题，具有广泛的应用。在众多方法中，最常见的方法是最大化模块度。尽管启发式模块度最大化算法设计理念和广泛采用，但很少返回最佳分区或类似分区。我们提出了一种专门的算法Bayan，它返回具有最优或接近最优分区保证的分区。Bayan算法的核心是一种分支限界方案，它解决了问题的整数规划公式以达到最优或近似最优的目的。我们证明Bayan在合成基准和真实网络节点标签的检索地面真实社区方面具有独特的准确性和稳定性，比其他21种算法快数倍，可以找到最优分区的实例。

    Community detection is a classic problem in network science with extensive applications in various fields. Among numerous approaches, the most common method is modularity maximization. Despite their design philosophy and wide adoption, heuristic modularity maximization algorithms rarely return an optimal partition or anything similar. We propose a specialized algorithm, Bayan, which returns partitions with a guarantee of either optimality or proximity to an optimal partition. At the core of the Bayan algorithm is a branch-and-cut scheme that solves an integer programming formulation of the problem to optimality or approximate it within a factor. We demonstrate Bayan's distinctive accuracy and stability over 21 other algorithms in retrieving ground-truth communities in synthetic benchmarks and node labels in real networks. Bayan is several times faster than open-source and commercial solvers for modularity maximization making it capable of finding optimal partitions for instances that c
    

