# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR](https://arxiv.org/abs/2402.03988) | 本文提出了REBORN，在无监督语音识别中使用基于强化学习的迭代训练来实现边界分割。通过交替训练分割模型和音素预测模型，实现了学习语音和文本之间的映射，解决了无监督情况下语音信号分段结构边界的挑战。 |

# 详细

[^1]: REBORN: 基于强化学习的迭代训练的无监督语音识别中的边界分割

    REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR

    [https://arxiv.org/abs/2402.03988](https://arxiv.org/abs/2402.03988)

    本文提出了REBORN，在无监督语音识别中使用基于强化学习的迭代训练来实现边界分割。通过交替训练分割模型和音素预测模型，实现了学习语音和文本之间的映射，解决了无监督情况下语音信号分段结构边界的挑战。

    

    无监督自动语音识别（ASR）旨在学习语音信号与其对应的文本转录之间的映射，而无需配对的语音-文本数据监督。语音信号中的单词/音素由一段长度可变且边界未知的语音信号表示，而这种分段结构使得在没有配对数据的情况下学习语音和文本之间的映射变得具有挑战性。本文提出了REBORN，基于强化学习的迭代训练的无监督语音识别中的边界分割。REBORN交替进行以下两个步骤：（1）训练一个能够预测语音信号中分段结构边界的分割模型，和（2）训练一个音素预测模型，其输入是由分割模型分割的分段结构，用于预测音素转录。由于没有用于训练分割模型的监督数据，我们使用强化学习来训练分割模型。

    Unsupervised automatic speech recognition (ASR) aims to learn the mapping between the speech signal and its corresponding textual transcription without the supervision of paired speech-text data. A word/phoneme in the speech signal is represented by a segment of speech signal with variable length and unknown boundary, and this segmental structure makes learning the mapping between speech and text challenging, especially without paired data. In this paper, we propose REBORN, Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR. REBORN alternates between (1) training a segmentation model that predicts the boundaries of the segmental structures in speech signals and (2) training the phoneme prediction model, whose input is a segmental structure segmented by the segmentation model, to predict a phoneme transcription. Since supervised data for training the segmentation model is not available, we use reinforcement learning to train the segmentation model t
    

