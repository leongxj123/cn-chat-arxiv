# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids.](http://arxiv.org/abs/2401.01145) | HAAQI-Net是一种适用于助听器用户的非侵入性神经音质评估模型，通过使用BLSTM和注意力机制，以及预训练的BEATs进行声学特征提取，能够快速且准确地预测音乐的HAAQI得分，相比传统方法具有更高的性能和更低的推理时间。 |

# 详细

[^1]: HAAQI-Net: 一种适用于助听器的非侵入性神经音质评估模型

    HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids. (arXiv:2401.01145v1 [eess.AS])

    [http://arxiv.org/abs/2401.01145](http://arxiv.org/abs/2401.01145)

    HAAQI-Net是一种适用于助听器用户的非侵入性神经音质评估模型，通过使用BLSTM和注意力机制，以及预训练的BEATs进行声学特征提取，能够快速且准确地预测音乐的HAAQI得分，相比传统方法具有更高的性能和更低的推理时间。

    

    本文介绍了HAAQI-Net，一种针对助听器用户定制的非侵入性深度学习音质评估模型。与传统方法如Hearing Aid Audio Quality Index (HAAQI) 不同，HAAQI-Net采用了带有注意力机制的双向长短期记忆网络(BLSTM)。该模型以评估的音乐样本和听力损失模式作为输入，生成预测的HAAQI得分。模型采用了预训练的来自音频变换器(BEATs)的双向编码器表示进行声学特征提取。通过将预测分数与真实分数进行比较，HAAQI-Net达到了0.9257的长期一致性相关(LCC)，0.9394的斯皮尔曼等级相关系数(SRCC)，和0.0080的均方误差(MSE)。值得注意的是，这种高性能伴随着推理时间的大幅减少：从62.52秒(HAAQI)减少到2.71秒(HAAQI-Net)，为助听器用户提供了高效的音质评估模型。

    This paper introduces HAAQI-Net, a non-intrusive deep learning model for music quality assessment tailored to hearing aid users. In contrast to traditional methods like the Hearing Aid Audio Quality Index (HAAQI), HAAQI-Net utilizes a Bidirectional Long Short-Term Memory (BLSTM) with attention. It takes an assessed music sample and a hearing loss pattern as input, generating a predicted HAAQI score. The model employs the pre-trained Bidirectional Encoder representation from Audio Transformers (BEATs) for acoustic feature extraction. Comparing predicted scores with ground truth, HAAQI-Net achieves a Longitudinal Concordance Correlation (LCC) of 0.9257, Spearman's Rank Correlation Coefficient (SRCC) of 0.9394, and Mean Squared Error (MSE) of 0.0080. Notably, this high performance comes with a substantial reduction in inference time: from 62.52 seconds (by HAAQI) to 2.71 seconds (by HAAQI-Net), serving as an efficient music quality assessment model for hearing aid users.
    

