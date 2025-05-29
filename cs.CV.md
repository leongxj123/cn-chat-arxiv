# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Meta Co-Training: Two Views are Better than One](https://arxiv.org/abs/2311.18083) | 元共训练通过在数据上构建不同的视角，并利用未标记数据进行共同训练，提高了半监督学习的性能。 |
| [^2] | [LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding](https://arxiv.org/abs/2311.15876) | RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。 |
| [^3] | [Variational Positive-incentive Noise: How Noise Benefits Models.](http://arxiv.org/abs/2306.07651) | 本文研究了如何通过正激励噪声框架下的随机噪声使经典模型受益，并提出了变分Pi-Noise，它可以在不改变原始模型结构的情况下增强和简化模型。 |

# 详细

[^1]: 元共训练：两种视角优于一种

    Meta Co-Training: Two Views are Better than One

    [https://arxiv.org/abs/2311.18083](https://arxiv.org/abs/2311.18083)

    元共训练通过在数据上构建不同的视角，并利用未标记数据进行共同训练，提高了半监督学习的性能。

    

    在许多实际的计算机视觉场景中，未标记的数据很多，但标签却稀缺且难以获得。因此，半监督学习利用未标记的数据提升监督分类器的性能已经在最近的文献中引起了重要的关注。其中一种主要的半监督算法是共训练。在共训练中，两种不同的模型利用数据的不同独立和足够的“视角”来共同进行更好的预测。在共训练过程中，每个模型在未标记的数据点上创建伪标签，用于改进另一个模型的性能。我们展示了在常见情况下，当独立视角不可用时，我们可以使用预训练模型来廉价地构建这些视角。在构建的视角上进行共训练可以提高性能，优于我们构建的任何单个视角，并且与半监督学习中的最新方法性能相当，但具有一些不可取之处。

    In many practical computer vision scenarios unlabeled data is plentiful, but labels are scarce and difficult to obtain. As a result, semi-supervised learning which leverages unlabeled data to boost the performance of supervised classifiers have received significant attention in recent literature. One major class of semi-supervised algorithms is co-training. In co-training two different models leverage different independent and sufficient "views" of the data to jointly make better predictions. During co-training each model creates pseudo labels on unlabeled points which are used to improve the other model. We show that in the common case when independent views are not available we can construct such views inexpensively using pre-trained models. Co-training on the constructed views yields a performance improvement over any of the individual views we construct and performance comparable with recent approaches in semi-supervised learning, but has some undesirable properties. To alleviate t
    
[^2]: LMM辅助的一致性嵌入下乳腺癌治疗目标分割

    LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding

    [https://arxiv.org/abs/2311.15876](https://arxiv.org/abs/2311.15876)

    RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。

    

    人工智能的最新进展深刻影响了医学领域，为降低临床工作量提供了工具。然而，大多数人工智能模型受限于执行单模式任务，与医学专业人员所使用的综合方法形成鲜明对比。为解决这一问题，本文介绍了RO-LMM，一个专为放射肿瘤学领域设计的多功能大型多模型（LMM）。该模型涵盖了临床工作流中的一系列任务，擅长临床报告摘要、放疗治疗计划建议和计划引导的目标体积分割。为了执行连续的临床任务，我们进一步提出了一种新颖的一致性嵌入微调（CEFTune）技术，提升了LMM对嘈杂输入的鲁棒性，同时保持了处理干净输入的能力，并将该概念转化为LMM驱动的分割框架，即一致性嵌入S。

    arXiv:2311.15876v2 Announce Type: replace-cross  Abstract: Recent advancements in Artificial Intelligence (AI) have profoundly influenced medical fields, by providing tools to reduce clinical workloads. However, most AI models are constrained to execute unimodal tasks, in stark contrast to the comprehensive approaches utilized by medical professionals. To address this, here we present RO-LMM, a multi-purpose large multimodal model (LMM) tailored for the field of radiation oncology. This model covers series of tasks within clinical workflow, adept at clinical report summarization, radiation treatment plan suggestion, and plan-guided target volume segmentation. In particular, to perform consecutive clinical tasks, we further present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LMM's robustness to noisy inputs while preserving the capability of handling clean inputs, and transform this concept into LMM-driven segmentation framework as Consistency Embedding S
    
[^3]: 变分激励噪声：噪声如何改进模型

    Variational Positive-incentive Noise: How Noise Benefits Models. (arXiv:2306.07651v1 [cs.LG])

    [http://arxiv.org/abs/2306.07651](http://arxiv.org/abs/2306.07651)

    本文研究了如何通过正激励噪声框架下的随机噪声使经典模型受益，并提出了变分Pi-Noise，它可以在不改变原始模型结构的情况下增强和简化模型。

    

    大量研究旨在减轻由于负面噪声的基本假设而导致的噪声影响。但是，一些现有的研究表明，这种假设并不总是成立的。本文研究了如何在正激励噪声（Pi-Noise）框架下通过随机噪声使经典模型受益。由于Pi-Noise的理想目标是难以实现的，我们提出了对其变分下界进行优化的变分Pi-Noise（VPN），通过变分推断，设计了一个VPN生成器来增强基础模型并简化基础模型的推断，而不改变基础模型的架构。由于基础模型和VPN生成器的独立设计， VPN生成器可以与大多数现有模型一起使用。从实验结果来看，所提出的VPN生成器可以改进基本模型。值得称赞的是，训练有素的变分VPN生成器更喜欢独立密集型噪声。（翻译有删减）

    A large number of works aim to alleviate the impact of noise due to an underlying conventional assumption of the negative role of noise. However, some existing works show that the assumption does not always hold. In this paper, we investigate how to benefit the classical models by random noise under the framework of Positive-incentive Noise (Pi-Noise). Since the ideal objective of Pi-Noise is intractable, we propose to optimize its variational bound instead, namely variational Pi-Noise (VPN). With the variational inference, a VPN generator implemented by neural networks is designed for enhancing base models and simplifying the inference of base models, without changing the architecture of base models. Benefiting from the independent design of base models and VPN generators, the VPN generator can work with most existing models. From the experiments, it is shown that the proposed VPN generator can improve the base models. It is appealing that the trained variational VPN generator prefers
    

