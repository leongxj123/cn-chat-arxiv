# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What if...?: Counterfactual Inception to Mitigate Hallucination Effects in Large Multimodal Models](https://arxiv.org/abs/2403.13513) | 本文引入了反事实启示（Counterfactual Inception）方法，通过将反事实思想植入到大型多模态模型（LMMs）中，可以减轻幻觉效应并提高模型的可信度。 |
| [^2] | [Connect Later: Improving Fine-tuning for Robustness with Targeted Augmentations](https://arxiv.org/abs/2402.03325) | 本文研究了在标记的源领域上训练的模型在部署到分布不同的目标领域时的泛化问题，并提出了一种名为连接延迟的方法，通过自监督预训练和定向增强方法来改善模型鲁棒性。 |
| [^3] | [A Good Feature Extractor Is All You Need for Weakly Supervised Pathology Slide Classification](https://arxiv.org/abs/2311.11772) | 在弱监督整个切片图像分类中，不同于常规认知的观念，研究发现省略染色标准化和图像增强并不会影响下游切片级别的分类性能，同时还能节省大量内存和计算资源。 |

# 详细

[^1]: 如果......会怎样？：反事实启示在大型多模态模型中减轻幻觉效应

    What if...?: Counterfactual Inception to Mitigate Hallucination Effects in Large Multimodal Models

    [https://arxiv.org/abs/2403.13513](https://arxiv.org/abs/2403.13513)

    本文引入了反事实启示（Counterfactual Inception）方法，通过将反事实思想植入到大型多模态模型（LMMs）中，可以减轻幻觉效应并提高模型的可信度。

    

    本文介绍了提高大型多模态模型（LMMs）在处理幻觉效应方面可靠性的方法，其中模型会生成不正确或无关的响应。没有额外的指导调整范式，我们引入了反事实启示，这是一种新颖的方法，通过精心选择的、不对齐的反事实关键词将反事实思想植入到LMMs中。该方法根植于反事实思维概念，这是一种认知过程，人类在其中考虑替代现实和结果。通过将这种类似人类的推理机制应用到LMMs中，我们旨在减少幻觉效应并提高模型的可信度。我们还提出了双模态验证过程（DVP），这是一个严格的框架，用于选择触发LMMs中反事实思维的最佳反事实关键词，同时考虑视觉和语言上下文。我们在各种LMMs上进行了大量实验

    arXiv:2403.13513v1 Announce Type: cross  Abstract: This paper presents a way of enhancing the reliability of Large Multimodal Models (LMMs) in addressing hallucination effects, where models generate incorrect or unrelated responses. Without additional instruction tuning paradigm, we introduce Counterfactual Inception, a novel method that implants counterfactual thoughts into LMMs using carefully chosen, misaligned counterfactual keywords. This method is grounded in the concept of counterfactual thinking, a cognitive process where humans consider alternative realities and outcomes. By applying this human-like reasoning mechanism to LMMs, we aim to reduce hallucination effects and improve the models' trustworthiness. We also propose Dual-modality Verification Process (DVP), a rigorous framework for selecting optimal counterfactual keywords to trigger counterfactual thinking into LMMs, concurrently considering visual and linguistic context. Our extensive experiments across various LMMs, i
    
[^2]: 连接延迟：利用定向增强方法提高鲁棒性的微调改进

    Connect Later: Improving Fine-tuning for Robustness with Targeted Augmentations

    [https://arxiv.org/abs/2402.03325](https://arxiv.org/abs/2402.03325)

    本文研究了在标记的源领域上训练的模型在部署到分布不同的目标领域时的泛化问题，并提出了一种名为连接延迟的方法，通过自监督预训练和定向增强方法来改善模型鲁棒性。

    

    在标记的源领域上训练的模型（例如野生动物相机陷阱的标记图像）通常在部署到分布不同的目标领域（例如新的相机陷阱位置的图像）时泛化能力较差。在存在未标记的目标数据的域适应设置中，自监督预训练（例如遮蔽自编码或对比学习）是缓解性能下降的一种有希望的方法。预训练可以通过将源领域和目标领域相连接的通用数据增强方法（例如遮蔽或剪裁）来提高分布不同的错误率，即使输入空间中的两个领域相差很远。本文通过真实任务展示了在预训练后进行标准微调并不能持续改善分布不同的错误率，相比在标记的源数据上从头训练。为了更好地利用预训练来应对分布转变，我们提出了连接延迟（Connect Later）：在使用通用增强方法进行预训练后，用基于对目标领域了解的定向增强方法进行微调。

    Models trained on a labeled source domain (e.g., labeled images from wildlife camera traps) often generalize poorly when deployed on an out-of-distribution (OOD) target domain (e.g., images from new camera trap locations). In the domain adaptation setting where unlabeled target data is available, self-supervised pretraining (e.g., masked autoencoding or contrastive learning) is a promising method to mitigate this performance drop. Pretraining improves OOD error when the generic data augmentations used (e.g., masking or cropping) connect the source and target domains, which may be far apart in the input space. In this paper, we show on real-world tasks that standard fine-tuning after pretraining does not consistently improve OOD error over simply training from scratch on labeled source data. To better leverage pretraining for distribution shifts, we propose Connect Later: after pretraining with generic augmentations, fine-tune with targeted augmentations designed with knowledge of the d
    
[^3]: 一个良好的特征提取器就是你在弱监督病理学切片分类中所需的一切

    A Good Feature Extractor Is All You Need for Weakly Supervised Pathology Slide Classification

    [https://arxiv.org/abs/2311.11772](https://arxiv.org/abs/2311.11772)

    在弱监督整个切片图像分类中，不同于常规认知的观念，研究发现省略染色标准化和图像增强并不会影响下游切片级别的分类性能，同时还能节省大量内存和计算资源。

    

    常规认为染色标准化是计算病理学流程中关键的预处理步骤。我们在弱监督的整个切片图像分类环境中对这一信念提出质疑，这一信念是由训练在多样化病理学数据集上进行自监督学习的强大特征提取器的出现所激励的。为此，我们对迄今为止公开可获得的病理学特征提取器进行了最全面的评估，涉及九个任务、五个数据集、三个下游架构和各种预处理设置中的8000多个训练运行。值得注意的是，我们发现忽略染色标准化和图像增强并不会损害下游切片级别的分类性能，同时还会在内存和计算上带来大量节省。通过使用一种新的评估指标，促进了相对下游性能的比较，我们确定了最好的公开可获得的提取器，并展示。

    arXiv:2311.11772v4 Announce Type: replace-cross  Abstract: Stain normalisation is thought to be a crucial preprocessing step in computational pathology pipelines. We question this belief in the context of weakly supervised whole slide image classification, motivated by the emergence of powerful feature extractors trained using self-supervised learning on diverse pathology datasets. To this end, we performed the most comprehensive evaluation of publicly available pathology feature extractors to date, involving more than 8,000 training runs across nine tasks, five datasets, three downstream architectures, and various preprocessing setups. Notably, we find that omitting stain normalisation and image augmentations does not compromise downstream slide-level classification performance, while incurring substantial savings in memory and compute. Using a new evaluation metric that facilitates relative downstream performance comparison, we identify the best publicly available extractors, and sho
    

