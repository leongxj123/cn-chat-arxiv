# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization](https://arxiv.org/abs/2403.16576) | 通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。 |
| [^2] | [A Survey of Generative AI for De Novo Drug Design: New Frontiers in Molecule and Protein Generation](https://arxiv.org/abs/2402.08703) | 这项综述提出了一个广义方法来驱动AI药物设计，重点关注小分子和蛋白质生成两个主要主题，介绍了各种子任务和应用，并比较了顶级模型的性能。 |

# 详细

[^1]: 通过直接基于能量偏好优化的抗原特异性抗体设计

    Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization

    [https://arxiv.org/abs/2403.16576](https://arxiv.org/abs/2403.16576)

    通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。

    

    抗体设计是一个至关重要的任务，对各种领域都有重要影响，如治疗和生物学，由于其错综复杂的性质，面临着相当大的挑战。在本文中，我们将抗原特异性抗体设计作为一个蛋白质序列-结构共设计问题，考虑了理性和功能性。利用一个预先训练的条件扩散模型，该模型联合建模抗体中互补决定区（CDR）的序列和结构，并结合了等变神经网络，我们提出了直接基于能量偏好优化的方法，以引导生成既具有合理结构又具有明显结合亲和力的抗体。我们的方法涉及使用残基级分解能量偏好对预先训练的扩散模型进行微调。此外，我们采用梯度手术来解决各种类型能量之间的冲突，例如吸引和斥

    arXiv:2403.16576v1 Announce Type: cross  Abstract: Antibody design, a crucial task with significant implications across various disciplines such as therapeutics and biology, presents considerable challenges due to its intricate nature. In this paper, we tackle antigen-specific antibody design as a protein sequence-structure co-design problem, considering both rationality and functionality. Leveraging a pre-trained conditional diffusion model that jointly models sequences and structures of complementarity-determining regions (CDR) in antibodies with equivariant neural networks, we propose direct energy-based preference optimization to guide the generation of antibodies with both rational structures and considerable binding affinities to given antigens. Our method involves fine-tuning the pre-trained diffusion model using a residue-level decomposed energy preference. Additionally, we employ gradient surgery to address conflicts between various types of energy, such as attraction and repu
    
[^2]: 生成式人工智能在全新药物设计中的应用概述：分子和蛋白质生成的新领域

    A Survey of Generative AI for De Novo Drug Design: New Frontiers in Molecule and Protein Generation

    [https://arxiv.org/abs/2402.08703](https://arxiv.org/abs/2402.08703)

    这项综述提出了一个广义方法来驱动AI药物设计，重点关注小分子和蛋白质生成两个主要主题，介绍了各种子任务和应用，并比较了顶级模型的性能。

    

    人工智能（AI）驱动的方法可以极大地改进历来代价高昂的药物设计过程，各种生成模型已经在广泛使用中。特别是针对全新药物设计的生成模型，专注于从零开始创建新的生物化合物，展示了一个有前景的未来方向。该领域的快速发展，加上药物设计过程的固有复杂性，为新研究人员进入创造了一个困难的局面。在这份综述中，我们将全新药物设计分为两个主要主题：小分子和蛋白质生成。在每个主题中，我们确定了各种子任务和应用，重点介绍了重要的数据集、基准和模型架构，并对顶级模型的性能进行了比较。我们采用了广义的方法来驱动AI药物设计，允许在每个子任务中进行各种方法的微观比较和宏观比较。

    arXiv:2402.08703v1 Announce Type: cross Abstract: Artificial intelligence (AI)-driven methods can vastly improve the historically costly drug design process, with various generative models already in widespread use. Generative models for de novo drug design, in particular, focus on the creation of novel biological compounds entirely from scratch, representing a promising future direction. Rapid development in the field, combined with the inherent complexity of the drug design process, creates a difficult landscape for new researchers to enter. In this survey, we organize de novo drug design into two overarching themes: small molecule and protein generation. Within each theme, we identify a variety of subtasks and applications, highlighting important datasets, benchmarks, and model architectures and comparing the performance of top models. We take a broad approach to AI-driven drug design, allowing for both micro-level comparisons of various methods within each subtask and macro-level o
    

