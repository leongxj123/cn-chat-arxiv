# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions.](http://arxiv.org/abs/2401.15296) | 本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。 |
| [^2] | [HCVP: Leveraging Hierarchical Contrastive Visual Prompt for Domain Generalization.](http://arxiv.org/abs/2401.09716) | HCVP是一种基于层次对比视觉提示的领域泛化方法，通过引导模型将不变特征与特定特征分离，提高了泛化性能。 |
| [^3] | [On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective.](http://arxiv.org/abs/2304.13836) | 本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。 |

# 详细

[^1]: 基于3D骨架的人员再识别：方法、设计、挑战和未来方向的综述

    A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions. (arXiv:2401.15296v1 [cs.CV])

    [http://arxiv.org/abs/2401.15296](http://arxiv.org/abs/2401.15296)

    本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。

    

    通过3D骨架进行人员再识别是一个重要的新兴研究领域，引起了模式识别社区的极大兴趣。近年来，针对骨架建模和特征学习中突出问题，已经提出了许多具有独特优势的基于3D骨架的人员再识别（SRID）方法。尽管近年来取得了一些进展，但据我们所知，目前还没有对这些研究及其挑战进行综合总结。因此，本文通过对当前SRID方法、模型设计、挑战和未来方向的系统调研，试图填补这一空白。具体而言，我们首先定义了SRID问题，并提出了一个SRID研究的分类体系，总结了常用的基准数据集、常用的模型架构，并对不同方法的特点进行了分析评价。然后，我们详细阐述了SRID模型的设计原则。

    Person re-identification via 3D skeletons is an important emerging research area that triggers great interest in the pattern recognition community. With distinctive advantages for many application scenarios, a great diversity of 3D skeleton based person re-identification (SRID) methods have been proposed in recent years, effectively addressing prominent problems in skeleton modeling and feature learning. Despite recent advances, to the best of our knowledge, little effort has been made to comprehensively summarize these studies and their challenges. In this paper, we attempt to fill this gap by providing a systematic survey on current SRID approaches, model designs, challenges, and future directions. Specifically, we first formulate the SRID problem, and propose a taxonomy of SRID research with a summary of benchmark datasets, commonly-used model architectures, and an analytical review of different methods' characteristics. Then, we elaborate on the design principles of SRID models fro
    
[^2]: HCVP: 基于层次对比视觉提示的领域泛化方法

    HCVP: Leveraging Hierarchical Contrastive Visual Prompt for Domain Generalization. (arXiv:2401.09716v1 [cs.CV])

    [http://arxiv.org/abs/2401.09716](http://arxiv.org/abs/2401.09716)

    HCVP是一种基于层次对比视觉提示的领域泛化方法，通过引导模型将不变特征与特定特征分离，提高了泛化性能。

    

    领域泛化（DG）旨在通过学习不变特征来创建在未知场景中表现出色的机器学习模型。然而，在DG中，将模型限制在固定结构或统一参数化中以包含不变特征的主流实践可能会不可避免地融合特定方面。这种方法难以对领域间变化进行细微区分，可能对某些领域存在偏见，从而阻碍了对域不变特征的精确学习。鉴于此，我们引入了一种新方法，旨在为模型提供领域级和任务特定的特征。该方法旨在更有效地引导模型将不变特征与特定特征分离，从而提高泛化性能。在领域泛化范式中，借鉴了视觉提示的新趋势，我们的工作引入了一种新颖的“HCVP”（层次对比视觉提示）方法。

    Domain Generalization (DG) endeavors to create machine learning models that excel in unseen scenarios by learning invariant features. In DG, the prevalent practice of constraining models to a fixed structure or uniform parameterization to encapsulate invariant features can inadvertently blend specific aspects. Such an approach struggles with nuanced differentiation of inter-domain variations and may exhibit bias towards certain domains, hindering the precise learning of domain-invariant features. Recognizing this, we introduce a novel method designed to supplement the model with domain-level and task-specific characteristics. This approach aims to guide the model in more effectively separating invariant features from specific characteristics, thereby boosting the generalization. Building on the emerging trend of visual prompts in the DG paradigm, our work introduces the novel \textbf{H}ierarchical \textbf{C}ontrastive \textbf{V}isual \textbf{P}rompt (HCVP) methodology. This represents 
    
[^3]: 论RemOve-And-Retrain的陷阱：数据处理不等式的视角

    On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective. (arXiv:2304.13836v1 [cs.LG])

    [http://arxiv.org/abs/2304.13836](http://arxiv.org/abs/2304.13836)

    本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。

    

    本文评估了RemOve-And-Retrain（ROAR）协议的可靠性，该协议用于测量特征重要性估计的性能。我们从理论背景和实证实验中发现，具有较少有关决策功能的信息的属性在ROAR基准测试中表现更好，与ROAR的原始目的相矛盾。这种现象也出现在最近提出的变体RemOve-And-Debias（ROAD）中，我们提出了ROAR归因度量中毛糙度偏差的一致趋势。我们的结果提醒人们不要盲目依赖ROAR的性能评估指标。

    This paper assesses the reliability of the RemOve-And-Retrain (ROAR) protocol, which is used to measure the performance of feature importance estimates. Our findings from the theoretical background and empirical experiments indicate that attributions that possess less information about the decision function can perform better in ROAR benchmarks, conflicting with the original purpose of ROAR. This phenomenon is also observed in the recently proposed variant RemOve-And-Debias (ROAD), and we propose a consistent trend of blurriness bias in ROAR attribution metrics. Our results caution against uncritical reliance on ROAR metrics.
    

