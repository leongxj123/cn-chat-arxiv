# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contrastive Approach to Prior Free Positive Unlabeled Learning](https://arxiv.org/abs/2402.06038) | 该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。 |

# 详细

[^1]: 免先验正无标（Positive Unlabeled）学习的对比方法

    Contrastive Approach to Prior Free Positive Unlabeled Learning

    [https://arxiv.org/abs/2402.06038](https://arxiv.org/abs/2402.06038)

    该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。

    

    正无标（Positive Unlabeled）学习是指在给定少量标记的正样本和一组未标记样本（可能是正例或负例）的情况下学习一个二分类器的任务。在本文中，我们提出了一种新颖的正无标学习框架，通过保证不变表示学习学习特征空间，并利用嵌入的浓度特性对未标记样本进行伪标签处理。总体而言，我们提出的方法在多个标准正无标基准数据集上轻松超越了现有的正无标学习方法，而不需要先验知识或类先验的估计。值得注意的是，我们的方法在标记数据稀缺的情况下仍然有效，而大多数正无标学习算法则失败。我们还提供了简单的理论分析来推动我们提出的算法，并为我们的方法建立了一般化保证。

    Positive Unlabeled (PU) learning refers to the task of learning a binary classifier given a few labeled positive samples, and a set of unlabeled samples (which could be positive or negative). In this paper, we propose a novel PU learning framework, that starts by learning a feature space through pretext-invariant representation learning and then applies pseudo-labeling to the unlabeled examples, leveraging the concentration property of the embeddings. Overall, our proposed approach handily outperforms state-of-the-art PU learning methods across several standard PU benchmark datasets, while not requiring a-priori knowledge or estimate of class prior. Remarkably, our method remains effective even when labeled data is scant, where most PU learning algorithms falter. We also provide simple theoretical analysis motivating our proposed algorithms and establish generalization guarantee for our approach.
    

