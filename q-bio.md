# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Brain-grounding of semantic vectors improves neural decoding of visual stimuli](https://arxiv.org/abs/2403.15176) | 提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。 |
| [^2] | [Brant-2: Foundation Model for Brain Signals](https://arxiv.org/abs/2402.10251) | Brant-2是脑信号领域最大的基础模型，相比于Brant，它不仅对数据变化和建模尺度具有稳健性，还能适用于更广泛范围的脑神经数据。 |

# 详细

[^1]: 语义向量的脑接地改善了神经解码视觉刺激

    Brain-grounding of semantic vectors improves neural decoding of visual stimuli

    [https://arxiv.org/abs/2403.15176](https://arxiv.org/abs/2403.15176)

    提出了一种表示学习框架，称为语义向量的脑接地，通过微调预训练的特征向量，使其更好地与人类大脑中视觉刺激的神经表示对齐。

    

    发展准确全面的算法来解码大脑内容是神经科学和脑机接口领域的一个长期目标。之前的研究已经证明了通过训练机器学习模型将大脑活动模式映射到一个语义向量表示的神经解码的可行性。为了解决这个问题，我们提出了一个表示学习框架，称为语义向量的脑接地，它对预训练的特征向量进行微调，以更好地与人类大脑中视觉刺激的神经表示对齐。

    arXiv:2403.15176v1 Announce Type: cross  Abstract: Developing algorithms for accurate and comprehensive neural decoding of mental contents is one of the long-cherished goals in the field of neuroscience and brain-machine interfaces. Previous studies have demonstrated the feasibility of neural decoding by training machine learning models to map brain activity patterns into a semantic vector representation of stimuli. These vectors, hereafter referred as pretrained feature vectors, are usually derived from semantic spaces based solely on image and/or text features and therefore they might have a totally different characteristics than how visual stimuli is represented in the human brain, resulting in limiting the capability of brain decoders to learn this mapping. To address this issue, we propose a representation learning framework, termed brain-grounding of semantic vectors, which fine-tunes pretrained feature vectors to better align with the neural representation of visual stimuli in t
    
[^2]: Brant-2：脑信号基础模型

    Brant-2: Foundation Model for Brain Signals

    [https://arxiv.org/abs/2402.10251](https://arxiv.org/abs/2402.10251)

    Brant-2是脑信号领域最大的基础模型，相比于Brant，它不仅对数据变化和建模尺度具有稳健性，还能适用于更广泛范围的脑神经数据。

    

    基础模型受益于在大量未标记数据上进行预训练，并且在少量标记数据的情况下能够在各种应用中表现出色。这种模型在分析脑信号方面特别有效，因为这一领域涵盖了众多应用场景，并且进行大规模注释是成本高昂的。在这项工作中，我们提出了脑信号领域最大的基础模型，Brant-2。与用于颅内神经信号的基础模型Brant相比，Brant-2不仅对数据变化和建模尺度表现出稳健性，而且可以应用于更广泛范围的脑神经数据。通过在大量任务上进行实验，我们展示了Brant-2对脑信号中各种应用场景的适应性。进一步分析揭示了Brant-2的可扩展性，验证了每个组件的有效性，并展示了我们模型保持的能力。

    arXiv:2402.10251v1 Announce Type: cross  Abstract: Foundational models benefit from pre-training on large amounts of unlabeled data and enable strong performance in a wide variety of applications with a small amount of labeled data. Such models can be particularly effective in analyzing brain signals, as this field encompasses numerous application scenarios, and it is costly to perform large-scale annotation. In this work, we present the largest foundation model in brain signals, Brant-2. Compared to Brant, a foundation model designed for intracranial neural signals, Brant-2 not only exhibits robustness towards data variations and modeling scales but also can be applied to a broader range of brain neural data. By experimenting on an extensive range of tasks, we demonstrate that Brant-2 is adaptive to various application scenarios in brain signals. Further analyses reveal the scalability of the Brant-2, validate each component's effectiveness, and showcase our model's ability to maintai
    

