# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization](https://arxiv.org/abs/2403.16576) | 通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。 |
| [^2] | [Transformer-based de novo peptide sequencing for data-independent acquisition mass spectrometry](https://arxiv.org/abs/2402.11363) | 这项研究提出了Casanovo-DIA，一种基于Transformer架构的深度学习模型，可用于从DIA质谱数据中解析肽段序列。 |
| [^3] | [A Survey of Generative AI for De Novo Drug Design: New Frontiers in Molecule and Protein Generation](https://arxiv.org/abs/2402.08703) | 这项综述提出了一个广义方法来驱动AI药物设计，重点关注小分子和蛋白质生成两个主要主题，介绍了各种子任务和应用，并比较了顶级模型的性能。 |
| [^4] | [Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis.](http://arxiv.org/abs/2401.11447) | 本研究利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期过敏性鼻炎亚卡激素皮下免疫治疗的管理提供了一种新的方法。 |

# 详细

[^1]: 通过直接基于能量偏好优化的抗原特异性抗体设计

    Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization

    [https://arxiv.org/abs/2403.16576](https://arxiv.org/abs/2403.16576)

    通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。

    

    抗体设计是一个至关重要的任务，对各种领域都有重要影响，如治疗和生物学，由于其错综复杂的性质，面临着相当大的挑战。在本文中，我们将抗原特异性抗体设计作为一个蛋白质序列-结构共设计问题，考虑了理性和功能性。利用一个预先训练的条件扩散模型，该模型联合建模抗体中互补决定区（CDR）的序列和结构，并结合了等变神经网络，我们提出了直接基于能量偏好优化的方法，以引导生成既具有合理结构又具有明显结合亲和力的抗体。我们的方法涉及使用残基级分解能量偏好对预先训练的扩散模型进行微调。此外，我们采用梯度手术来解决各种类型能量之间的冲突，例如吸引和斥

    arXiv:2403.16576v1 Announce Type: cross  Abstract: Antibody design, a crucial task with significant implications across various disciplines such as therapeutics and biology, presents considerable challenges due to its intricate nature. In this paper, we tackle antigen-specific antibody design as a protein sequence-structure co-design problem, considering both rationality and functionality. Leveraging a pre-trained conditional diffusion model that jointly models sequences and structures of complementarity-determining regions (CDR) in antibodies with equivariant neural networks, we propose direct energy-based preference optimization to guide the generation of antibodies with both rational structures and considerable binding affinities to given antigens. Our method involves fine-tuning the pre-trained diffusion model using a residue-level decomposed energy preference. Additionally, we employ gradient surgery to address conflicts between various types of energy, such as attraction and repu
    
[^2]: 基于Transformer的全新肽段测序技术用于数据无偏向采集质谱

    Transformer-based de novo peptide sequencing for data-independent acquisition mass spectrometry

    [https://arxiv.org/abs/2402.11363](https://arxiv.org/abs/2402.11363)

    这项研究提出了Casanovo-DIA，一种基于Transformer架构的深度学习模型，可用于从DIA质谱数据中解析肽段序列。

    

    串联质谱（MS/MS）作为全面分析生物样本中蛋白质含量的主要高通量技术，一直是推动蛋白质组学发展的基石。近年来，在数据无偏向采集（DIA）策略方面取得了实质性进展，有助于对前体离子进行公正和非靶向碎裂。由于其固有的高多重性特性，DIA生成的MS/MS谱图造成了巨大障碍。每个谱图都包含来自多个前体肽的碎裂产物离子。这种复杂性特别在全新肽段/蛋白质测序中构成了一个尖锐挑战，当前方法无法解决多重性难题。本文介绍了Casanovo-DIA，这是一个基于Transformer架构的深度学习模型，可以从DIA质谱数据中解析肽段序列。

    arXiv:2402.11363v1 Announce Type: cross  Abstract: Tandem mass spectrometry (MS/MS) stands as the predominant high-throughput technique for comprehensively analyzing protein content within biological samples. This methodology is a cornerstone driving the advancement of proteomics. In recent years, substantial strides have been made in Data-Independent Acquisition (DIA) strategies, facilitating impartial and non-targeted fragmentation of precursor ions. The DIA-generated MS/MS spectra present a formidable obstacle due to their inherent high multiplexing nature. Each spectrum encapsulates fragmented product ions originating from multiple precursor peptides. This intricacy poses a particularly acute challenge in de novo peptide/protein sequencing, where current methods are ill-equipped to address the multiplexing conundrum. In this paper, we introduce Casanovo-DIA, a deep-learning model based on transformer architecture. It deciphers peptide sequences from DIA mass spectrometry data. Our 
    
[^3]: 生成式人工智能在全新药物设计中的应用概述：分子和蛋白质生成的新领域

    A Survey of Generative AI for De Novo Drug Design: New Frontiers in Molecule and Protein Generation

    [https://arxiv.org/abs/2402.08703](https://arxiv.org/abs/2402.08703)

    这项综述提出了一个广义方法来驱动AI药物设计，重点关注小分子和蛋白质生成两个主要主题，介绍了各种子任务和应用，并比较了顶级模型的性能。

    

    人工智能（AI）驱动的方法可以极大地改进历来代价高昂的药物设计过程，各种生成模型已经在广泛使用中。特别是针对全新药物设计的生成模型，专注于从零开始创建新的生物化合物，展示了一个有前景的未来方向。该领域的快速发展，加上药物设计过程的固有复杂性，为新研究人员进入创造了一个困难的局面。在这份综述中，我们将全新药物设计分为两个主要主题：小分子和蛋白质生成。在每个主题中，我们确定了各种子任务和应用，重点介绍了重要的数据集、基准和模型架构，并对顶级模型的性能进行了比较。我们采用了广义的方法来驱动AI药物设计，允许在每个子任务中进行各种方法的微观比较和宏观比较。

    arXiv:2402.08703v1 Announce Type: cross Abstract: Artificial intelligence (AI)-driven methods can vastly improve the historically costly drug design process, with various generative models already in widespread use. Generative models for de novo drug design, in particular, focus on the creation of novel biological compounds entirely from scratch, representing a promising future direction. Rapid development in the field, combined with the inherent complexity of the drug design process, creates a difficult landscape for new researchers to enter. In this survey, we organize de novo drug design into two overarching themes: small molecule and protein generation. Within each theme, we identify a variety of subtasks and applications, highlighting important datasets, benchmarks, and model architectures and comparing the performance of top models. We take a broad approach to AI-driven drug design, allowing for both micro-level comparisons of various methods within each subtask and macro-level o
    
[^4]: 预测过敏性鼻炎亚卡激素皮下免疫治疗中患者依从性的序列模型

    Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis. (arXiv:2401.11447v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.11447](http://arxiv.org/abs/2401.11447)

    本研究利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期过敏性鼻炎亚卡激素皮下免疫治疗的管理提供了一种新的方法。

    

    目标：皮下免疫治疗(SCIT)是过敏性鼻炎的长效因果治疗。如何提高患者对变应原免疫治疗(AIT)的依从性以最大化治疗效果，在AIT管理中起着至关重要的作用。本研究旨在利用新颖的机器学习模型，准确预测患者的非依从风险和相关的系统症状评分，为长期AIT的管理提供一种新的方法。方法：本研究开发和分析了两种模型，序列潜在行为者-评论家模型(SLAC)和长短期记忆模型(LSTM)，并基于评分和依从性预测能力进行评估。结果：在排除第一时间步的偏倚样本后，SLAC模型的预测依从准确率为60%-72%，而LSTM模型的准确率为66%-84%，根据时间步长的不同而变化。SLAC模型的均方根误差(RMSE)范围在0.93到2.22之间，而LSTM模型的RMSE范围在...

    Objective: Subcutaneous Immunotherapy (SCIT) is the long-lasting causal treatment of allergic rhinitis. How to enhance the adherence of patients to maximize the benefit of allergen immunotherapy (AIT) plays a crucial role in the management of AIT. This study aims to leverage novel machine learning models to precisely predict the risk of non-adherence of patients and related systematic symptom scores, to provide a novel approach in the management of long-term AIT.  Methods: The research develops and analyzes two models, Sequential Latent Actor-Critic (SLAC) and Long Short-Term Memory (LSTM), evaluating them based on scoring and adherence prediction capabilities.  Results: Excluding the biased samples at the first time step, the predictive adherence accuracy of the SLAC models is from $60\,\%$ to $72\%$, and for LSTM models, it is $66\,\%$ to $84\,\%$, varying according to the time steps. The range of Root Mean Square Error (RMSE) for SLAC models is between $0.93$ and $2.22$, while for L
    

