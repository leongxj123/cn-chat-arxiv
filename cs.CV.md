# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlexCap: Generating Rich, Localized, and Flexible Captions in Images](https://arxiv.org/abs/2403.12026) | FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。 |
| [^2] | [Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) | 评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标 |
| [^3] | [Out-of-distribution detection using normalizing flows on the data manifold.](http://arxiv.org/abs/2308.13792) | 利用正则化流在低维数据流形上进行外域检测，通过估计密度和测量与流形的距离来判断外域数据，有效提高外域检测的准确性。 |

# 详细

[^1]: FlexCap：在图像中生成丰富、本地化和灵活的标题

    FlexCap: Generating Rich, Localized, and Flexible Captions in Images

    [https://arxiv.org/abs/2403.12026](https://arxiv.org/abs/2403.12026)

    FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。

    

    我们介绍了一种多功能的$\textit{灵活字幕}$视觉-语言模型（VLM），能够生成长度不同的特定区域描述。该模型FlexCap经过训练，可为输入的边界框生成长度条件的字幕，从而可以控制其输出的信息密度，描述范围从简洁的对象标签到详细的字幕。为了实现这一点，我们从带字幕的图像开始创建了大规模的图像区域描述训练数据集。这种灵活的字幕功能有几个宝贵的应用。首先，FlexCap在Visual Genome数据集上的密集字幕任务中表现出优越性能。其次，可以通过采用FlexCap生成本地化描述作为大型语言模型的输入来构建视觉问答（VQA）系统。由此产生的系统在许多VQ上实现了最新技术的零样本性能。

    arXiv:2403.12026v1 Announce Type: cross  Abstract: We introduce a versatile $\textit{flexible-captioning}$ vision-language model (VLM) capable of generating region-specific descriptions of varying lengths. The model, FlexCap, is trained to produce length-conditioned captions for input bounding boxes, and this allows control over the information density of its output, with descriptions ranging from concise object labels to detailed captions. To achieve this we create large-scale training datasets of image region descriptions of varying length, starting from captioned images. This flexible-captioning capability has several valuable applications.   First, FlexCap demonstrates superior performance in dense captioning tasks on the Visual Genome dataset. Second, a visual question answering (VQA) system can be built by employing FlexCap to generate localized descriptions as inputs to a large language model. The resulting system achieves state-of-the-art zero-shot performance on a number of VQ
    
[^2]: 评估文本到图像合成：图像质量度量的调查与分类

    Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics

    [https://arxiv.org/abs/2403.11821](https://arxiv.org/abs/2403.11821)

    评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标

    

    最近，通过利用语言和视觉结合的基础模型，推动了文本到图像合成方面的进展。这些模型在互联网或其他大规模数据库中的海量文本-图像对上进行了预训练。随着对高质量图像生成的需求转向确保文本与图像之间的内容对齐，已开发了新颖的评估度量标准，旨在模拟人类判断。因此，研究人员开始收集具有越来越复杂注释的数据集，以研究视觉语言模型的组成性及其作为文本与图像内容组成对齐质量度量的其纳入。在这项工作中，我们全面介绍了现有的文本到图像评估指标，并提出了一个新的分类法来对这些指标进行分类。我们还审查了经常采用的文本-图像基准数据集

    arXiv:2403.11821v1 Announce Type: cross  Abstract: Recent advances in text-to-image synthesis have been enabled by exploiting a combination of language and vision through foundation models. These models are pre-trained on tremendous amounts of text-image pairs sourced from the World Wide Web or other large-scale databases. As the demand for high-quality image generation shifts towards ensuring content alignment between text and image, novel evaluation metrics have been developed with the aim of mimicking human judgments. Thus, researchers have started to collect datasets with increasingly complex annotations to study the compositionality of vision-language models and their incorporation as a quality measure of compositional alignment between text and image contents. In this work, we provide a comprehensive overview of existing text-to-image evaluation metrics and propose a new taxonomy for categorizing these metrics. We also review frequently adopted text-image benchmark datasets befor
    
[^3]: 利用数据流形上的正则化流进行外域检测

    Out-of-distribution detection using normalizing flows on the data manifold. (arXiv:2308.13792v1 [cs.LG])

    [http://arxiv.org/abs/2308.13792](http://arxiv.org/abs/2308.13792)

    利用正则化流在低维数据流形上进行外域检测，通过估计密度和测量与流形的距离来判断外域数据，有效提高外域检测的准确性。

    

    外域检测的一种常见方法是估计基础数据分布，为外域数据分配较低的可能性值。正则化流是基于可能性的生成模型，通过保持维度的可逆变换提供可计算的密度估计。传统的正则化流在外域检测中容易失败，因为基于可能性的模型面临着维度诅咒的问题。根据流形假设，现实世界的数据通常位于低维流形上。本研究调查了使用正则化流进行外域检测时的流形学习效果。我们通过在低维流形上估计密度，并结合测量与流形的距离作为外域检测的标准。然而，单独使用它们对于这个任务是不足够的。广泛的实验证明了流形学习对外域检测的有效性。

    A common approach for out-of-distribution detection involves estimating an underlying data distribution, which assigns a lower likelihood value to out-of-distribution data. Normalizing flows are likelihood-based generative models providing a tractable density estimation via dimension-preserving invertible transformations. Conventional normalizing flows are prone to fail in out-of-distribution detection, because of the well-known curse of dimensionality problem of the likelihood-based models. According to the manifold hypothesis, real-world data often lie on a low-dimensional manifold. This study investigates the effect of manifold learning using normalizing flows on out-of-distribution detection. We proceed by estimating the density on a low-dimensional manifold, coupled with measuring the distance from the manifold, as criteria for out-of-distribution detection. However, individually, each of them is insufficient for this task. The extensive experimental results show that manifold lea
    

