# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UDEEP: Edge-based Computer Vision for In-Situ Underwater Crayfish and Plastic Detection.](http://arxiv.org/abs/2401.06157) | UDEEP是一个基于边缘计算机视觉的平台，可以帮助解决入侵信号龙虾和废弃塑料对水生生态系统的挑战。 |
| [^2] | [DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing.](http://arxiv.org/abs/2310.08785) | 本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。 |
| [^3] | [Cross-Modal Retrieval for Motion and Text via MildTriple Loss.](http://arxiv.org/abs/2305.04195) | 本论文提出了一个创新模型，使用MildTriple Loss捕捉长期依赖并模拟跨模态人类动作序列与文本检索任务，具有重要的应用价值。 |

# 详细

[^1]: UDEEP: 基于边缘的水下信号龙虾和塑料检测的计算机视觉

    UDEEP: Edge-based Computer Vision for In-Situ Underwater Crayfish and Plastic Detection. (arXiv:2401.06157v1 [cs.CV])

    [http://arxiv.org/abs/2401.06157](http://arxiv.org/abs/2401.06157)

    UDEEP是一个基于边缘计算机视觉的平台，可以帮助解决入侵信号龙虾和废弃塑料对水生生态系统的挑战。

    

    入侵的信号龙虾对生态系统造成了不利影响。它们传播了对英国唯一的本地白爪龙虾致命的真菌型龙虾瘟疫病(Aphanomyces astaci)。入侵的信号龙虾广泛挖掘洞穴，破坏栖息地，侵蚀河岸并对水质产生不利影响，同时竞争本地物种的资源并导致本地种群下降。此外，污染也使白爪龙虾更加容易受到损害，其种群在英国某些地区下降超过90％，使其极易濒临灭绝。为了保护水生生态系统，解决入侵物种和废弃塑料对英国河流生态系统的挑战至关重要。UDEEP平台可以通过实时分类信号龙虾和塑料碎片，充当环境监测的关键角色。

    Invasive signal crayfish have a detrimental impact on ecosystems. They spread the fungal-type crayfish plague disease (Aphanomyces astaci) that is lethal to the native white clawed crayfish, the only native crayfish species in Britain. Invasive signal crayfish extensively burrow, causing habitat destruction, erosion of river banks and adverse changes in water quality, while also competing with native species for resources and leading to declines in native populations. Moreover, pollution exacerbates the vulnerability of White-clawed crayfish, with their populations declining by over 90% in certain English counties, making them highly susceptible to extinction. To safeguard aquatic ecosystems, it is imperative to address the challenges posed by invasive species and discarded plastics in the United Kingdom's river ecosystem's. The UDEEP platform can play a crucial role in environmental monitoring by performing on-the-fly classification of Signal crayfish and plastic debris while leveragi
    
[^2]: DeltaSpace:一种用于灵活文本引导图像编辑的语义对齐特征空间

    DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing. (arXiv:2310.08785v1 [cs.CV])

    [http://arxiv.org/abs/2310.08785](http://arxiv.org/abs/2310.08785)

    本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。

    

    文本引导图像编辑面临着训练和推理灵活性的重大挑战。许多文献通过收集大量标注的图像-文本对来从头开始训练文本条件生成模型，这既昂贵又低效。然后，一些利用预训练的视觉语言模型的方法出现了，以避免数据收集，但它们仍然受到基于每个文本提示的优化或推理时的超参数调整的限制。为了解决这些问题，我们调查和确定了一个特定的空间，称为CLIP DeltaSpace，在这个空间中，两个图像的CLIP视觉特征差异与其对应的文本描述的CLIP文本特征差异在语义上是对齐的。基于DeltaSpace，我们提出了一个新颖的框架DeltaEdit，在训练阶段将CLIP视觉特征差异映射到生成模型的潜在空间方向，并从CLIP预测潜在空间方向。

    Text-guided image editing faces significant challenges to training and inference flexibility. Much literature collects large amounts of annotated image-text pairs to train text-conditioned generative models from scratch, which is expensive and not efficient. After that, some approaches that leverage pre-trained vision-language models are put forward to avoid data collection, but they are also limited by either per text-prompt optimization or inference-time hyper-parameters tuning. To address these issues, we investigate and identify a specific space, referred to as CLIP DeltaSpace, where the CLIP visual feature difference of two images is semantically aligned with the CLIP textual feature difference of their corresponding text descriptions. Based on DeltaSpace, we propose a novel framework called DeltaEdit, which maps the CLIP visual feature differences to the latent space directions of a generative model during the training phase, and predicts the latent space directions from the CLIP
    
[^3]: MildTriple Loss模型下的运动和文本跨模态检索

    Cross-Modal Retrieval for Motion and Text via MildTriple Loss. (arXiv:2305.04195v1 [cs.CV])

    [http://arxiv.org/abs/2305.04195](http://arxiv.org/abs/2305.04195)

    本论文提出了一个创新模型，使用MildTriple Loss捕捉长期依赖并模拟跨模态人类动作序列与文本检索任务，具有重要的应用价值。

    

    跨模态检索已成为计算机视觉和自然语言处理中的重要研究课题，随着图像文本和视频文本检索技术的进步。尽管在虚拟现实等广泛应用中具有重要价值，但人类动作序列与文本之间的跨模态检索尚未引起足够的关注。这个任务存在一些挑战，包括对两种语言的共同建模，要求从文本中理解以人为中心的信息，并从三维人体运动序列中学习行为特征。以往的运动数据建模主要依赖于自回归特征提取器，这可能会遗忘以前的信息，而我们提出了一种创新模型，其中包括简单而强大的基于变换器的运动和文本编码器，可以从两种不同的模态中学习表示并捕捉长期依赖

    Cross-modal retrieval has become a prominent research topic in computer vision and natural language processing with advances made in image-text and video-text retrieval technologies. However, cross-modal retrieval between human motion sequences and text has not garnered sufficient attention despite the extensive application value it holds, such as aiding virtual reality applications in better understanding users' actions and language. This task presents several challenges, including joint modeling of the two modalities, demanding the understanding of person-centered information from text, and learning behavior features from 3D human motion sequences. Previous work on motion data modeling mainly relied on autoregressive feature extractors that may forget previous information, while we propose an innovative model that includes simple yet powerful transformer-based motion and text encoders, which can learn representations from the two different modalities and capture long-term dependencie
    

