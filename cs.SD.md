# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reduce, Reuse, Recycle: Is Perturbed Data better than Other Language augmentation for Low Resource Self-Supervised Speech Models.](http://arxiv.org/abs/2309.12763) | 使用音频增强为低资源自我监督语音模型的预训练提出一种有效的方法，并且综合增强（噪声/音高）是最佳的增强策略，超过了重音和语言知识转移。 |

# 详细

[^1]: 减少、复用、回收：与其他语言增强相比，被扰动数据对低资源自我监督语音模型更好吗？

    Reduce, Reuse, Recycle: Is Perturbed Data better than Other Language augmentation for Low Resource Self-Supervised Speech Models. (arXiv:2309.12763v1 [eess.AS])

    [http://arxiv.org/abs/2309.12763](http://arxiv.org/abs/2309.12763)

    使用音频增强为低资源自我监督语音模型的预训练提出一种有效的方法，并且综合增强（噪声/音高）是最佳的增强策略，超过了重音和语言知识转移。

    

    自我监督表示学习（SSRL）已经改善了下游音素识别的性能，相对于受监督的模型。训练SSRL模型需要大量的预训练数据，这对于低资源语言是一个挑战。一种常用的方法是从其他语言中转移知识。相反，我们提出使用音频增强在低资源条件下预训练SSRL模型，并评估下游任务的音素识别。我们对增强技术进行了系统比较，包括音高变化、噪声添加、有重音的目标语音和其他语言的语音。我们发现综合增强（噪声/音高）是最好的增强策略，超过了重音和语言知识转移。我们比较了不同数量和类型的预训练数据的性能。我们考察了增强数据的缩放因子，以达到与预训练目标域语音模型相当的性能。我们的发现是...

    Self-supervised representation learning (SSRL) has improved the performance on downstream phoneme recognition versus supervised models. Training SSRL models requires a large amount of pre-training data and this poses a challenge for low resource languages. A common approach is transferring knowledge from other languages. Instead, we propose to use audio augmentation to pre-train SSRL models in a low resource condition and evaluate phoneme recognition as downstream task. We performed a systematic comparison of augmentation techniques, namely: pitch variation, noise addition, accented target-language speech and other language speech. We found combined augmentations (noise/pitch) was the best augmentation strategy outperforming accent and language knowledge transfer. We compared the performance with various quantities and types of pre-training data. We examined the scaling factor of augmented data to achieve equivalent performance to models pre-trained with target domain speech. Our findi
    

