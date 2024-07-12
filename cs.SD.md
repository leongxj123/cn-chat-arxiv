# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Paralingual are Paralinguistic Representations? A Case Study in Speech Emotion Recognition](https://rss.arxiv.org/abs/2402.01579) | 本研究通过比较研究五种预训练模型，评估了针对社交语言任务进行预训练的模型在多语言情境下对语音情感识别的效果。结果表明，TRILLsson模型能够有效地捕捉语音数据中的社交语言特征，提升了语音情感识别的性能。 |
| [^2] | [Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach.](http://arxiv.org/abs/2401.10747) | 本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。 |
| [^3] | [Sumformer: A Linear-Complexity Alternative to Self-Attention for Speech Recognition.](http://arxiv.org/abs/2307.07421) | Sumformer提出了一种线性时间代替自注意力的方法，用总结混合来处理语音识别任务，可以在保持准确性的同时降低训练和推理时间。 |

# 详细

[^1]: 多重语言环境下语音情感识别的跨语言预训练模型研究

    How Paralingual are Paralinguistic Representations? A Case Study in Speech Emotion Recognition

    [https://rss.arxiv.org/abs/2402.01579](https://rss.arxiv.org/abs/2402.01579)

    本研究通过比较研究五种预训练模型，评估了针对社交语言任务进行预训练的模型在多语言情境下对语音情感识别的效果。结果表明，TRILLsson模型能够有效地捕捉语音数据中的社交语言特征，提升了语音情感识别的性能。

    

    预训练模型（PTM）在语音情感识别（SER）领域取得了巨大进展。最近的研究利用各种PTM表示作为SER下游模型的输入特征。针对社交语言任务进行预训练的PTM在SER领域取得了最先进的性能。然而，这些PTM还没有在多语言环境下进行SER评估，且只涉及英语。因此，我们通过对五种PTM（TRILLsson、wav2vec2、XLS-R、x-vector、Whisper）进行全面比较研究，评估社交语言PTM（TRILLsson）在多种语言情境下对SER的效果。TRILLsson的表示在所有PTM中达到了最佳表现。这表明TRILLsson能够有效捕捉语音数据中的各种社交语言特征，从而提供更好的SER。

    Pre-trained Models (PTMs) have facilitated substantial progress in the field of Speech Emotion Recognition (SER). SER is an area with applications ranging from HumanComputer Interaction to Healthcare. Recent studies have leveraged various PTM representations as input features for downstream models for SER. PTM specifically pre-trained for paralinguistic tasks have obtained state-of-the-art (SOTA) performance for SER. However, such PTM haven't been evaluated for SER in multilingual settings and experimented only with English. So, we fill this gap, by performing a comprehensive comparative study of five PTMs (TRILLsson, wav2vec2, XLS-R, x-vector, Whisper) for assessing the effectiveness of paralingual PTM (TRILLsson) for SER across multiple languages. Representations from TRILLsson achieved the best performance among all the PTMs. This demonstrates that TRILLsson is able to effectively capture the various paralinguistic features from speech data for better SER. We also show that downstre
    
[^2]: 缺失模态下的多模态情感分析:一种知识迁移方法

    Multimodal Sentiment Analysis with Missing Modality: A Knowledge-Transfer Approach. (arXiv:2401.10747v1 [cs.SD])

    [http://arxiv.org/abs/2401.10747](http://arxiv.org/abs/2401.10747)

    本文提出了一种知识迁移方法，用于在缺失模态下进行多模态情感分析。通过翻译不同模态之间的内容以重构缺失的音频模态，并利用跨模态注意机制进行情感预测，实验证明了该方法在多个数据集上表现出显著的改进和与完整多模态监督方法相媲美的效果。

    

    多模态情感分析旨在通过视觉、语言和声音线索来识别个体表达的情绪。然而，现有研究大多假设在训练和测试过程中所有模态都是可用的，这使得它们的算法容易受到缺失模态的影响。在本文中，我们提出了一种新颖的知识迁移网络，用于在不同模态之间进行翻译，以重构缺失的音频模态。此外，我们还开发了一种跨模态注意机制，以保留重构和观察到的模态的最大信息，用于情感预测。在三个公开数据集上进行的大量实验证明了相对于基线算法的显著改进，并实现了与具有完整多模态监督的先前方法相媲美的结果。

    Multimodal sentiment analysis aims to identify the emotions expressed by individuals through visual, language, and acoustic cues. However, most of the existing research efforts assume that all modalities are available during both training and testing, making their algorithms susceptible to the missing modality scenario. In this paper, we propose a novel knowledge-transfer network to translate between different modalities to reconstruct the missing audio modalities. Moreover, we develop a cross-modality attention mechanism to retain the maximal information of the reconstructed and observed modalities for sentiment prediction. Extensive experiments on three publicly available datasets demonstrate significant improvements over baselines and achieve comparable results to the previous methods with complete multi-modality supervision.
    
[^3]: Sumformer: 一种用于语音识别的线性复杂度代替自注意力的方法

    Sumformer: A Linear-Complexity Alternative to Self-Attention for Speech Recognition. (arXiv:2307.07421v1 [cs.CL])

    [http://arxiv.org/abs/2307.07421](http://arxiv.org/abs/2307.07421)

    Sumformer提出了一种线性时间代替自注意力的方法，用总结混合来处理语音识别任务，可以在保持准确性的同时降低训练和推理时间。

    

    现代语音识别系统依赖于自注意力。然而，使用自注意力进行令牌混合的计算复杂度与语音语句的长度呈二次关系，导致推理、训练和内存占用速度变慢。虽然已经开发出了比自注意力更便宜的替代方法，但很难保证达到相同的准确性水平。实际上，经过训练的语音识别器的自注意力权重在时间上呈全局平均化的形式。因此，本文提出了一种用于语音识别的线性时间替代自注意力的方法。它用所有时间步长的向量的平均值来总结整个语句。然后将这个单一的总结与特定时间的信息结合起来。我们将这种方法称为“总结混合”。在最先进的ASR模型中引入总结混合，可以在降低训练和推理时间多达27%的同时，保持或超过先前的语音识别性能水平。

    Modern speech recognition systems rely on self-attention. Unfortunately, token mixing with self-attention takes quadratic time in the length of the speech utterance, slowing down inference as well as training and increasing memory consumption. Cheaper alternatives to self-attention for ASR have been developed, but fail to consistently reach the same level of accuracy. In practice, however, the self-attention weights of trained speech recognizers take the form of a global average over time. This paper, therefore, proposes a linear-time alternative to self-attention for speech recognition. It summarises a whole utterance with the mean over vectors for all time steps. This single summary is then combined with time-specific information. We call this method ``Summary Mixing''. Introducing Summary Mixing in state-of-the-art ASR models makes it feasible to preserve or exceed previous speech recognition performance while lowering the training and inference times by up to 27% and reducing the m
    

