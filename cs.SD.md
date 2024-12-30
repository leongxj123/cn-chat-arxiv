# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training.](http://arxiv.org/abs/2306.00107) | 提出了一个带有大规模自监督训练的音乐理解模型MERT，利用了教师模型并采用了一种优于传统的语音和音频方法的组合方式。 |
| [^2] | [Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification.](http://arxiv.org/abs/2305.14032) | 本研究提出了一种新的通过在音频数据上进行对比学习的方法，在呼吸音分类任务中取得了最先进的性能表现。 |

# 详细

[^1]: MERT:带有大规模自监督训练的声学音乐理解模型

    MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. (arXiv:2306.00107v1 [cs.SD])

    [http://arxiv.org/abs/2306.00107](http://arxiv.org/abs/2306.00107)

    提出了一个带有大规模自监督训练的音乐理解模型MERT，利用了教师模型并采用了一种优于传统的语音和音频方法的组合方式。

    

    自监督学习（SSL）最近在视觉、文本和语音领域中已被证明是训练通用模型的一种很有前景的范例，对于跨越音乐领域的应用，尤其是对于调性和音高这样的特殊音乐知识的建模颇具挑战性。为了解决这一问题，我们提出了一个基于大规模自监督训练的声学音乐理解模型，即MERT。在我们的探索中，我们确定了更优秀的教师模型组合，这种组合方法在性能方面优于传统的语音和音频方法。

    Self-supervised learning (SSL) has recently emerged as a promising paradigm for training generalisable models on large-scale data in the fields of vision, text, and speech. Although SSL has been proven effective in speech and audio, its application to music audio has yet to be thoroughly explored. This is primarily due to the distinctive challenges associated with modelling musical knowledge, particularly its tonal and pitched characteristics of music. To address this research gap, we propose an acoustic Music undERstanding model with large-scale self-supervised Training (MERT), which incorporates teacher models to provide pseudo labels in the masked language modelling (MLM) style acoustic pre-training. In our exploration, we identified a superior combination of teacher models, which outperforms conventional speech and audio approaches in terms of performance. This combination includes an acoustic teacher based on Residual Vector Quantization - Variational AutoEncoder (RVQ-VAE) and a m
    
[^2]: 带有音频光谱变换器的 Patch-Mix 对比学习在呼吸音分类中的应用

    Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification. (arXiv:2305.14032v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2305.14032](http://arxiv.org/abs/2305.14032)

    本研究提出了一种新的通过在音频数据上进行对比学习的方法，在呼吸音分类任务中取得了最先进的性能表现。

    

    呼吸声包含早期诊断致命肺部疾病的重要信息。自 COVID-19 疫情以来，基于电子听诊器的无接触医疗越来越受关注。为此，开发了先进的深度学习模型来诊断肺部疾病；然而，由于医学数据的稀缺，仍然存在挑战。本研究证明了在大规模视觉和音频数据集上预训练的模型可以推广到呼吸音分类任务。此外，我们引入了一种简单的 Patch-Mix 数据增强方法，通过随机混合不同样本之间的补丁，与 Audio Spectrogram Transformer (AST) 相结合。我们进一步提出了一种新颖而有效的 Patch-Mix 对比学习方法，以区分潜在空间中的混合表示。我们的方法在 ICBHI 数据集上取得了最先进的性能，优于先前的最高得分 4.08%。

    Respiratory sound contains crucial information for the early diagnosis of fatal lung diseases. Since the COVID-19 pandemic, there has been a growing interest in contact-free medical care based on electronic stethoscopes. To this end, cutting-edge deep learning models have been developed to diagnose lung diseases; however, it is still challenging due to the scarcity of medical data. In this study, we demonstrate that the pretrained model on large-scale visual and audio datasets can be generalized to the respiratory sound classification task. In addition, we introduce a straightforward Patch-Mix augmentation, which randomly mixes patches between different samples, with Audio Spectrogram Transformer (AST). We further propose a novel and effective Patch-Mix Contrastive Learning to distinguish the mixed representations in the latent space. Our method achieves state-of-the-art performance on the ICBHI dataset, outperforming the prior leading score by an improvement of 4.08%.
    

