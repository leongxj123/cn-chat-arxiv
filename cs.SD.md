# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A vector quantized masked autoencoder for audiovisual speech emotion recognition.](http://arxiv.org/abs/2305.03568) | 本文提出了一种特别为音视频言语自监督表示学习设计的矢量量化MAE模型，采用了基于离散音频和视觉言语表示的自监督范式，并在标准情感音视频言语数据集上取得了较好的效果。 |

# 详细

[^1]: 一种用于音视频言语情感识别的矢量量化掩码自编码器

    A vector quantized masked autoencoder for audiovisual speech emotion recognition. (arXiv:2305.03568v1 [cs.SD])

    [http://arxiv.org/abs/2305.03568](http://arxiv.org/abs/2305.03568)

    本文提出了一种特别为音视频言语自监督表示学习设计的矢量量化MAE模型，采用了基于离散音频和视觉言语表示的自监督范式，并在标准情感音视频言语数据集上取得了较好的效果。

    

    尽管全面监督模型已被证明对于音视频言语情感识别（SER）非常有效，但标记数据的有限性仍然是该领域的主要挑战。为了解决这个问题，自监督学习方法，如掩码自编码器（MAEs），已成为潜在解决方案。本文提出了一种特别为音视频言语自监督表示学习设计的矢量量化MAE模型（VQ-MAE-AV）。与现有的依赖于原始音视频言语数据处理的多模态MAEs不同，该方法采用了基于两个预先训练的矢量量化变分自编码器学习的离散音频和视觉言语表示的自监督范式。实验结果表明，该方法在VoxCeleb2数据库上进行预训练，并在标准情感音视频言语数据集上进行微调，优于现有的音视频SER方法。

    While fully-supervised models have been shown to be effective for audiovisual speech emotion recognition (SER), the limited availability of labeled data remains a major challenge in the field. To address this issue, self-supervised learning approaches, such as masked autoencoders (MAEs), have gained popularity as potential solutions. In this paper, we propose the VQ-MAE-AV model, a vector quantized MAE specifically designed for audiovisual speech self-supervised representation learning. Unlike existing multimodal MAEs that rely on the processing of the raw audiovisual speech data, the proposed method employs a self-supervised paradigm based on discrete audio and visual speech representations learned by two pre-trained vector quantized variational autoencoders. Experimental results show that the proposed approach, which is pre-trained on the VoxCeleb2 database and fine-tuned on standard emotional audiovisual speech datasets, outperforms the state-of-the-art audiovisual SER methods.
    

