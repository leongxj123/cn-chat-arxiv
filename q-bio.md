# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Efficient Sleep Staging with Synthetic Time Series Pretraining](https://arxiv.org/abs/2403.08592) | 通过预测合成时间序列的频率内容进行预训练，实现了在有限数据和少受试者情况下超越完全监督学习的方法 |
| [^2] | [Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function](https://arxiv.org/abs/2402.17131) | 本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展 |

# 详细

[^1]: 用合成时间序列预训练实现高效的睡眠分期

    Data-Efficient Sleep Staging with Synthetic Time Series Pretraining

    [https://arxiv.org/abs/2403.08592](https://arxiv.org/abs/2403.08592)

    通过预测合成时间序列的频率内容进行预训练，实现了在有限数据和少受试者情况下超越完全监督学习的方法

    

    分析脑电图（EEG）时间序列可能具有挑战性，特别是在深度神经网络中，由于人类受试者之间的大量变异和通常规模较小的数据集。为了解决这些挑战，提出了各种策略，例如自监督学习，但它们通常依赖于广泛的实证数据集。受计算机视觉最新进展的启发，我们提出了一种预训练任务，称为“频率预训练”，通过预测随机生成的合成时间序列的频率内容来为睡眠分期预训练神经网络。我们的实验表明，我们的方法在有限数据和少受试者的情况下优于完全监督学习，并在许多受试者的情境中表现相匹配。此外，我们的结果强调了频率信息对于睡眠分期评分的相关性，同时表明深度神经网络利用了超出频率信息的信息。

    arXiv:2403.08592v1 Announce Type: new  Abstract: Analyzing electroencephalographic (EEG) time series can be challenging, especially with deep neural networks, due to the large variability among human subjects and often small datasets. To address these challenges, various strategies, such as self-supervised learning, have been suggested, but they typically rely on extensive empirical datasets. Inspired by recent advances in computer vision, we propose a pretraining task termed "frequency pretraining" to pretrain a neural network for sleep staging by predicting the frequency content of randomly generated synthetic time series. Our experiments demonstrate that our method surpasses fully supervised learning in scenarios with limited data and few subjects, and matches its performance in regimes with many subjects. Furthermore, our results underline the relevance of frequency information for sleep stage scoring, while also demonstrating that deep neural networks utilize information beyond fr
    
[^2]: 使用Transformer和RNN在经过训练的新损失函数下预测哺乳动物蛋白质中的O-GlcNAcylation位点

    Predicting O-GlcNAcylation Sites in Mammalian Proteins with Transformers and RNNs Trained with a New Loss Function

    [https://arxiv.org/abs/2402.17131](https://arxiv.org/abs/2402.17131)

    本研究提出了一种新的损失函数，称为加权焦点可微MCC，用于改善分类模型的性能，并在预测哺乳动物蛋白质中的O-GlcNAcylation位点方面取得了进展

    

    糖基化是一种蛋白质修饰，在功能和结构上起着多种重要作用。O-GlcNAcylation是糖基化的一种亚型，有潜力成为治疗的重要靶点，但在2023年之前尚未有可靠预测O-GlcNAcylation位点的方法；2021年的一篇评论正确指出已发表的模型不足，并且未能泛化。此外，许多模型已不再可用。2023年，一篇具有F$_1$分数36.17%和MCC分数34.57%的大型数据集上的显着更好的RNN模型被发表。本文首次试图通过Transformer编码器提高这些指标。尽管Transformer在该数据集上表现出色，但其性能仍不及先前发表的RNN。然后我们创建了一种新的损失函数，称为加权焦点可微MCC，以提高分类模型的性能。

    arXiv:2402.17131v1 Announce Type: new  Abstract: Glycosylation, a protein modification, has multiple essential functional and structural roles. O-GlcNAcylation, a subtype of glycosylation, has the potential to be an important target for therapeutics, but methods to reliably predict O-GlcNAcylation sites had not been available until 2023; a 2021 review correctly noted that published models were insufficient and failed to generalize. Moreover, many are no longer usable. In 2023, a considerably better RNN model with an F$_1$ score of 36.17% and an MCC of 34.57% on a large dataset was published. This article first sought to improve these metrics using transformer encoders. While transformers displayed high performance on this dataset, their performance was inferior to that of the previously published RNN. We then created a new loss function, which we call the weighted focal differentiable MCC, to improve the performance of classification models. RNN models trained with this new function di
    

