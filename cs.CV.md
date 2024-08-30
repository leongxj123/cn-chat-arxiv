# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Humans Beat Deep Networks at Recognizing Objects in Unusual Poses, Given Enough Time](https://arxiv.org/abs/2402.03973) | 人类在识别不寻常姿势中的物体上表现优于深度网络，当给予足够时间时。然而，随着图像曝光时间的限制，人类的表现降至深度网络的水平，这暗示人类在识别不寻常姿势中的物体时需要额外的心理过程。此外，人类与网络之间的错误模式也存在不同。因此，我们需要进一步研究，以提高计算机视觉系统的鲁棒性水平。 |
| [^2] | [DiffiT: Diffusion Vision Transformers for Image Generation](https://arxiv.org/abs/2312.02139) | DiffiT是一种新的模型，结合了Vision Transformer和扩散模型的优势，在图像生成中表现出色，特别是通过引入细粒度去噪控制和时间依赖的多头自注意力机制，实现了高保真图像的生成。 |
| [^3] | [Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition](https://arxiv.org/abs/2011.08388) | 本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。 |

# 详细

[^1]: 给予足够时间，人类在识别不寻常姿势中的物体上击败了深度网络

    Humans Beat Deep Networks at Recognizing Objects in Unusual Poses, Given Enough Time

    [https://arxiv.org/abs/2402.03973](https://arxiv.org/abs/2402.03973)

    人类在识别不寻常姿势中的物体上表现优于深度网络，当给予足够时间时。然而，随着图像曝光时间的限制，人类的表现降至深度网络的水平，这暗示人类在识别不寻常姿势中的物体时需要额外的心理过程。此外，人类与网络之间的错误模式也存在不同。因此，我们需要进一步研究，以提高计算机视觉系统的鲁棒性水平。

    

    深度学习在几个物体识别基准上正在缩小与人类的差距。本文在涉及从不寻常视角观察物体的挑战性图像中对这一差距进行了研究。我们发现人类在识别不寻常姿势中的物体时表现出色，与先进的预训练网络（EfficientNet、SWAG、ViT、SWIN、BEiT、ConvNext）相比，这些网络在此情况下普遍脆弱。值得注意的是，随着我们限制图像曝光时间，人类的表现下降到深度网络的水平，这表明人类在识别不寻常姿势中的物体时需要额外的心理过程（需要额外的时间）。最后，我们分析了人类与网络的错误模式，发现即使在限制时间的情况下，人类与前馈深度网络也有不同。我们得出结论，需要更多的工作将计算机视觉系统带到人类视觉系统的鲁棒性水平。理解在外部情况下发生的心理过程的本质是必要的。

    Deep learning is closing the gap with humans on several object recognition benchmarks. Here we investigate this gap in the context of challenging images where objects are seen from unusual viewpoints. We find that humans excel at recognizing objects in unusual poses, in contrast with state-of-the-art pretrained networks (EfficientNet, SWAG, ViT, SWIN, BEiT, ConvNext) which are systematically brittle in this condition. Remarkably, as we limit image exposure time, human performance degrades to the level of deep networks, suggesting that additional mental processes (requiring additional time) take place when humans identify objects in unusual poses. Finally, our analysis of error patterns of humans vs. networks reveals that even time-limited humans are dissimilar to feed-forward deep networks. We conclude that more work is needed to bring computer vision systems to the level of robustness of the human visual system. Understanding the nature of the mental processes taking place during extr
    
[^2]: DiffiT: 用于图像生成的扩散视觉Transformer模型

    DiffiT: Diffusion Vision Transformers for Image Generation

    [https://arxiv.org/abs/2312.02139](https://arxiv.org/abs/2312.02139)

    DiffiT是一种新的模型，结合了Vision Transformer和扩散模型的优势，在图像生成中表现出色，特别是通过引入细粒度去噪控制和时间依赖的多头自注意力机制，实现了高保真图像的生成。

    

    具有强大表现力和高样本质量的扩散模型在生成领域取得了最先进的性能。开创性的视觉Transformer（ViT）展现了强大的建模能力和可扩展性，特别适用于识别任务。本文研究了ViTs在基于扩散的生成学习中的有效性，并提出了一个新模型，称为Diffusion Vision Transformers（DiffiT）。具体地，我们提出了一种用于对去噪过程进行细粒度控制的方法，并引入了时间依赖的多头自注意力（TMSA）机制。DiffiT在生成高保真图像方面非常有效，参数效率也显著提高。我们还提出了基于潜空间和图像空间的DiffiT模型，并在不同分辨率的各种类别条件和非条件综合任务上展现了最先进的性能。潜空间DiffiT模型达到

    arXiv:2312.02139v2 Announce Type: replace-cross  Abstract: Diffusion models with their powerful expressivity and high sample quality have achieved State-Of-The-Art (SOTA) performance in the generative domain. The pioneering Vision Transformer (ViT) has also demonstrated strong modeling capabilities and scalability, especially for recognition tasks. In this paper, we study the effectiveness of ViTs in diffusion-based generative learning and propose a new model denoted as Diffusion Vision Transformers (DiffiT). Specifically, we propose a methodology for finegrained control of the denoising process and introduce the Time-dependant Multihead Self Attention (TMSA) mechanism. DiffiT is surprisingly effective in generating high-fidelity images with significantly better parameter efficiency. We also propose latent and image space DiffiT models and show SOTA performance on a variety of class-conditional and unconditional synthesis tasks at different resolutions. The Latent DiffiT model achieves
    
[^3]: 基于领域自适应的可解释图像情绪识别，并利用面部表情识别

    Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition

    [https://arxiv.org/abs/2011.08388](https://arxiv.org/abs/2011.08388)

    本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。

    

    本文提出了一种领域自适应技术，用于识别包含面部和非面部物体以及非人类组件的通用图像中的情绪。它解决了图像情绪识别（IER）中预训练模型和良好注释数据集的不足挑战。首先，提出了一种基于深度学习的面部情绪识别（FER）系统，将给定的面部图像分类为离散情绪类别。然后，提出了一种图像识别系统，将提出的FER系统适应于利用领域自适应识别图像所传达的情绪。它将通用图像分类为“快乐”，“悲伤”，“仇恨”和“愤怒”类别。还提出了一种新颖的解释性方法，称为分而治之的Shap（DnCShap），用于解释情绪识别中高度相关的视觉特征。

    A domain adaptation technique has been proposed in this paper to identify the emotions in generic images containing facial & non-facial objects and non-human components. It addresses the challenge of the insufficient availability of pre-trained models and well-annotated datasets for image emotion recognition (IER). It starts with proposing a facial emotion recognition (FER) system and then moves on to adapting it for image emotion recognition. First, a deep-learning-based FER system has been proposed that classifies a given facial image into discrete emotion classes. Further, an image recognition system has been proposed that adapts the proposed FER system to recognize the emotions portrayed by images using domain adaptation. It classifies the generic images into 'happy,' 'sad,' 'hate,' and 'anger' classes. A novel interpretability approach, Divide and Conquer based Shap (DnCShap), has also been proposed to interpret the highly relevant visual features for emotion recognition. The prop
    

