# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Speech emotion recognition from voice messages recorded in the wild](https://arxiv.org/abs/2403.02167) | 使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。 |

# 详细

[^1]: 从野外录制的语音消息中识别语音情感

    Speech emotion recognition from voice messages recorded in the wild

    [https://arxiv.org/abs/2403.02167](https://arxiv.org/abs/2403.02167)

    使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。

    

    用于语音情感识别（SER）的情感数据集通常包含表演或引发的语音，限制了它们在现实场景中的适用性。在这项工作中，我们使用了Emotional Voice Messages（EMOVOME）数据库，其中包括来自100名西班牙语使用者在消息应用中的自发语音消息，由专家和非专家标注者以连续和离散的情感进行标记。我们使用了eGeMAPS特征、基于Transformer的模型以及它们的组合来创建讲话者无关的SER模型。我们将结果与参考数据库进行了比较，并分析了标注者和性别公平性的影响。预训练的Unispeech-L模型及其与eGeMAPS的组合取得了最佳结果，在3类valence和arousal预测中分别获得了61.64%和55.57%的Unweighted Accuracy（UA），比基线模型提高了10%。对于情感类别，获得了42.58%的UA。EMOVOME表现不佳。

    arXiv:2403.02167v1 Announce Type: cross  Abstract: Emotion datasets used for Speech Emotion Recognition (SER) often contain acted or elicited speech, limiting their applicability in real-world scenarios. In this work, we used the Emotional Voice Messages (EMOVOME) database, including spontaneous voice messages from conversations of 100 Spanish speakers on a messaging app, labeled in continuous and discrete emotions by expert and non-expert annotators. We created speaker-independent SER models using the eGeMAPS features, transformer-based models and their combination. We compared the results with reference databases and analyzed the influence of annotators and gender fairness. The pre-trained Unispeech-L model and its combination with eGeMAPS achieved the highest results, with 61.64% and 55.57% Unweighted Accuracy (UA) for 3-class valence and arousal prediction respectively, a 10% improvement over baseline models. For the emotion categories, 42.58% UA was obtained. EMOVOME performed low
    

