# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Comprehensive Evaluation of Augmentations for Robust OOD Self-Supervised Contrastive Phonocardiogram Representation Learning](https://arxiv.org/abs/2312.00502) | 本研究通过对比自监督学习应用于1D心音图样本中异常检测，进行了广泛的音频增强方法比较评估和在多个数据集上训练分类器的研究。 |
| [^2] | [JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation.](http://arxiv.org/abs/2310.19180) | JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。 |

# 详细

[^1]: 对稳健的OOD自监督对比心音图表示学习增强方法的全面评估

    A Comprehensive Evaluation of Augmentations for Robust OOD Self-Supervised Contrastive Phonocardiogram Representation Learning

    [https://arxiv.org/abs/2312.00502](https://arxiv.org/abs/2312.00502)

    本研究通过对比自监督学习应用于1D心音图样本中异常检测，进行了广泛的音频增强方法比较评估和在多个数据集上训练分类器的研究。

    

    尽管近年来深度学习模型的研究活动有所增加，但在医学等多个现实世界环境中，这些模型尚未被广泛接受。高质量标记数据的短缺经常阻碍了开发稳健且具有一般性的模型，当面临新收集的超出分布（OOD）数据集时，这些模型不会因效果下降而受损。对比自监督学习（SSL）为标记数据稀缺性提供了潜在解决方案，因为它利用未标记数据增加模型的效能和稳健性。本研究中，我们提出将对比SSL应用于检测1D心音图（PCG）样本中的异常，通过学习信号的广义表示。具体来说，我们进行了一项广泛的比较评估，涉及多种基于音频的增强方法，评估了在不同下游任务的多个数据集上训练的分类器，最终

    arXiv:2312.00502v2 Announce Type: replace  Abstract: Despite the recent increase in research activity, deep-learning models have not yet been widely accepted in several real-world settings, such as medicine. The shortage of high-quality annotated data often hinders the development of robust and generalizable models, which do not suffer from degraded effectiveness when presented with newly-collected, out-of-distribution (OOD) datasets. Contrastive Self-Supervised Learning (SSL) offers a potential solution to labeled data scarcity, as it takes advantage of unlabeled data to increase model effectiveness and robustness. In this research, we propose applying contrastive SSL for detecting abnormalities in 1D phonocardiogram (PCG) samples by learning a generalized representation of the signal. Specifically, we perform an extensive comparative evaluation of a wide range of audio-based augmentations, evaluate trained classifiers on multiple datasets across different downstream tasks, and finall
    
[^2]: JEN-1 Composer: 一个用于高保真多音轨音乐生成的统一框架

    JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation. (arXiv:2310.19180v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2310.19180](http://arxiv.org/abs/2310.19180)

    JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。

    

    随着生成式人工智能的快速发展，从零开始生成音乐的文本到音乐合成任务已成为一个有前景的方向。然而，对于多音轨生成的更细粒度控制仍然是一个挑战。现有模型具有较强的原始生成能力，但缺乏以可控的方式单独组成和组合多音轨的灵活性，这与人类作曲家的典型工作流程不同。为了解决这个问题，我们提出了JEN-1 Composer，一个统一的框架，通过一个模型高效地建模多音轨音乐的边缘、条件和联合分布。JEN-1 Composer框架能够无缝地整合任何基于扩散的音乐生成系统，例如Jen-1，增强其多功能多音轨音乐生成能力。我们引入了一种课程训练策略，以逐步指导模型从单音轨生成到灵活的生成过程。

    With rapid advances in generative artificial intelligence, the text-to-music synthesis task has emerged as a promising direction for music generation from scratch. However, finer-grained control over multi-track generation remains an open challenge. Existing models exhibit strong raw generation capability but lack the flexibility to compose separate tracks and combine them in a controllable manner, differing from typical workflows of human composers. To address this issue, we propose JEN-1 Composer, a unified framework to efficiently model marginal, conditional, and joint distributions over multi-track music via a single model. JEN-1 Composer framework exhibits the capacity to seamlessly incorporate any diffusion-based music generation system, \textit{e.g.} Jen-1, enhancing its capacity for versatile multi-track music generation. We introduce a curriculum training strategy aimed at incrementally instructing the model in the transition from single-track generation to the flexible genera
    

