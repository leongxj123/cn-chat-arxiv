# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models.](http://arxiv.org/abs/2401.13537) | 本文提出了一种称为遮蔽粒子建模（MPM）的自监督方法，用于学习高能物理科学数据中无序输入的通用表示。该方法通过预训练学习置换不变的函数，在构建适用于多种任务的高能物理基础模型方面具有潜力。 |

# 详细

[^1]: 基于集合的遮蔽粒子建模：走向自监督高能物理基础模型

    Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models. (arXiv:2401.13537v1 [hep-ph])

    [http://arxiv.org/abs/2401.13537](http://arxiv.org/abs/2401.13537)

    本文提出了一种称为遮蔽粒子建模（MPM）的自监督方法，用于学习高能物理科学数据中无序输入的通用表示。该方法通过预训练学习置换不变的函数，在构建适用于多种任务的高能物理基础模型方面具有潜力。

    

    本文提出了一种称为"遮蔽粒子建模"（MPM）的自监督方法，用于学习高能物理（HEP）科学数据中无序输入的通用、可转移和可重用表示。这项工作提供了一种新颖的方案，通过基于遮蔽建模的预训练来学习集合上的置换不变函数。更一般地，这项工作在构建可以通过自监督学习进行通用预训练并稍后精调用于各种下游任务的HEP大型基础模型方面迈出了一步。在MPM中，集合中的粒子被遮蔽，训练的目标是恢复它们的身份，身份由预训练的向量量化变分自动编码器的离散化标记表示定义。我们研究了该方法在对撞机物理实验中高能喷注样本上的有效性，包括离散化、置换不变性和排序的影响。

    We propose \textit{masked particle modeling} (MPM) as a self-supervised method for learning generic, transferable, and reusable representations on unordered sets of inputs for use in high energy physics (HEP) scientific data. This work provides a novel scheme to perform masked modeling based pre-training to learn permutation invariant functions on sets. More generally, this work provides a step towards building large foundation models for HEP that can be generically pre-trained with self-supervised learning and later fine-tuned for a variety of down-stream tasks. In MPM, particles in a set are masked and the training objective is to recover their identity, as defined by a discretized token representation of a pre-trained vector quantized variational autoencoder. We study the efficacy of the method in samples of high energy jets at collider physics experiments, including studies on the impact of discretization, permutation invariance, and ordering. We also study the fine-tuning capabili
    

