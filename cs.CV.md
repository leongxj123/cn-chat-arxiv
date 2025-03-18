# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |
| [^3] | [Valley: Video Assistant with Large Language model Enhanced abilitY.](http://arxiv.org/abs/2306.07207) | 本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。 |
| [^4] | [RViDeformer: Efficient Raw Video Denoising Transformer with a Larger Benchmark Dataset.](http://arxiv.org/abs/2305.00767) | 本文提出了RViDeformer原始视频去噪变换器及其配套数据集ReCRVD，其中利用高低ISO设置重新捕捉现有视频以构建噪声-清晰对，同时探索了非本地时空依赖关系的解决方案。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    
[^3]: Valley: 大型语言模型增强视频助手

    Valley: Video Assistant with Large Language model Enhanced abilitY. (arXiv:2306.07207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.07207](http://arxiv.org/abs/2306.07207)

    本文介绍了一个名为Valley的视频助手，它是一个以大型语言模型增强的多模态基础模型，能够在一个通用框架内理解视频、图像和语言。

    

    大型语言模型(LLMs)以其卓越的会话能力，在各种应用中表现出色，并成为强大的AI助手。鉴于此，一个直观的问题是：我们能否利用LLMs的能力构建多模态的视觉应用AI助手？最近，已经开发了几个多模态模型来实现这个目的。它们通常预先训练一个适应模块来对齐视觉编码器和语言模型的语义，然后在指令跟随数据上进行微调。然而，尽管这个流程在图像和语言理解方面取得了成功，在视频和语言理解方面的有效性还没有得到广泛探索。在本文中，我们旨在开发一个能够在一个通用框架内理解视频、图像和语言的新型多模态基础模型。为了实现这一目标，我们引入了Valley，一个以大型语言模型增强的视频助手。

    Large language models (LLMs), with their remarkable conversational capabilities, have demonstrated impressive performance across various applications and have emerged as formidable AI assistants. In view of this, it raises an intuitive question: Can we harness the power of LLMs to build multimodal AI assistants for visual applications? Recently, several multi-modal models have been developed for this purpose. They typically pre-train an adaptation module to align the semantics of the vision encoder and language model, followed by fine-tuning on instruction-following data. However, despite the success of this pipeline in image and language understanding, its effectiveness in joint video and language understanding has not been widely explored. In this paper, we aim to develop a novel multi-modal foundation model capable of comprehending video, image, and language within a general framework. To achieve this goal, we introduce Valley, a Video Assistant with Large Language model Enhanced ab
    
[^4]: RViDeformer：具有更大基准数据集的高效原始视频去噪变换器

    RViDeformer: Efficient Raw Video Denoising Transformer with a Larger Benchmark Dataset. (arXiv:2305.00767v1 [cs.CV])

    [http://arxiv.org/abs/2305.00767](http://arxiv.org/abs/2305.00767)

    本文提出了RViDeformer原始视频去噪变换器及其配套数据集ReCRVD，其中利用高低ISO设置重新捕捉现有视频以构建噪声-清晰对，同时探索了非本地时空依赖关系的解决方案。

    

    近年来，由于与成像过程的一致性和原始领域中成熟的噪声建模，原始视频去噪引起了越来越多的关注。然而，仍存在两个问题阻碍了去噪性能。首先，对于受控的原始视频去噪来说，没有具有真实运动的大型数据集，因为为真实动态场景捕捉噪声和清晰帧是困难的。为了解决这个问题，我们提出重新捕捉以高低ISO设置显示的现有高分辨率视频以构建噪声-清晰配对帧。这样，我们构建一个视频去噪数据集（命名为ReCRVD），其中包括120组噪声-清晰视频，其ISO值从1600到25600不等。其次，虽然非本地时空关注对于去噪有益，但它通常导致沉重的计算成本。为此，我们提出了一种高效的原始视频去噪变换器网络（RViDeformer），它探索了短距离和长距离相关性。具体而言，我们提出了一个空间变换器，用于处理本地视觉特征以减少空间冗余并加速计算。此外，为了解决长程噪声相关性，我们采用了一种新的时间变换器网络，同时模型化非本地时空依赖关系。

    In recent years, raw video denoising has garnered increased attention due to the consistency with the imaging process and well-studied noise modeling in the raw domain. However, two problems still hinder the denoising performance. Firstly, there is no large dataset with realistic motions for supervised raw video denoising, as capturing noisy and clean frames for real dynamic scenes is difficult. To address this, we propose recapturing existing high-resolution videos displayed on a 4K screen with high-low ISO settings to construct noisy-clean paired frames. In this way, we construct a video denoising dataset (named as ReCRVD) with 120 groups of noisy-clean videos, whose ISO values ranging from 1600 to 25600. Secondly, while non-local temporal-spatial attention is beneficial for denoising, it often leads to heavy computation costs. We propose an efficient raw video denoising transformer network (RViDeformer) that explores both short and long-distance correlations. Specifically, we propos
    

