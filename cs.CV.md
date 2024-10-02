# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training](https://arxiv.org/abs/2404.00226) | 本研究是首次利用视觉问答（VQA）进行多模态预训练，专注于引导模型学习所需病理特征，并提出了一种无需额外专家注释的问题-答案对设计方法，以及一种准文本特征转换器模块。 |
| [^2] | [ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding](https://arxiv.org/abs/2403.15004) | ParFormer提出了并行局部全局标记混合器和卷积注意力补丁嵌入，优化了特征提取能力，在图像分类和对象识别等任务中表现优于CNN和最先进的Transformer架构。 |
| [^3] | [How Far Are We from Intelligent Visual Deductive Reasoning?](https://arxiv.org/abs/2403.04732) | 目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。 |
| [^4] | [Shrub of a thousand faces: an individual segmentation from satellite images using deep learning](https://arxiv.org/abs/2401.17985) | 该研究提出了一种新方法，利用深度学习模型从卫星图像中个别描绘位于西班牙内华达山脉林线上的杜松灌木。研究通过整合遥感影像和实例分割模型，解决了检测和描绘自然对象复杂生长模式的挑战。 |
| [^5] | [Mitigating Biases with Diverse Ensembles and Diffusion Models](https://arxiv.org/abs/2311.16176) | 通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。 |
| [^6] | [Efficient generative adversarial networks using linear additive-attention Transformers.](http://arxiv.org/abs/2401.09596) | 这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。 |

# 详细

[^1]: 想要的设计：利用视觉问答进行多模态预训练

    Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training

    [https://arxiv.org/abs/2404.00226](https://arxiv.org/abs/2404.00226)

    本研究是首次利用视觉问答（VQA）进行多模态预训练，专注于引导模型学习所需病理特征，并提出了一种无需额外专家注释的问题-答案对设计方法，以及一种准文本特征转换器模块。

    

    多模态预训练在医疗领域展示了其潜力，从成对的医疗报告中学习医学视觉表示。然而，许多预训练任务需要临床医生额外的注释，大多数任务未能明确引导模型学习不同病理特征。据我们所知，我们是第一个利用视觉问答（VQA）进行多模态预训练的团队，以引导框架专注于目标病理特征。在这项工作中，我们利用医疗报告中的描述设计了与不同疾病相关的多粒度问题-答案对，这有助于框架在预训练中无需专家额外的注释。我们还提出了一种新颖的预训练框架，其中包括一种准文本特征转换器模块，旨在通过将视觉特征转换到接近文本领域的准文本空间来辅助预训练过程。

    arXiv:2404.00226v1 Announce Type: cross  Abstract: Multimodal pre-training demonstrates its potential in the medical domain, which learns medical visual representations from paired medical reports. However, many pre-training tasks require extra annotations from clinicians, and most of them fail to explicitly guide the model to learn the desired features of different pathologies. To the best of our knowledge, we are the first to utilize Visual Question Answering (VQA) for multimodal pre-training to guide the framework focusing on targeted pathological features. In this work, we leverage descriptions in medical reports to design multi-granular question-answer pairs associated with different diseases, which assist the framework in pre-training without requiring extra annotations from experts. We also propose a novel pre-training framework with a quasi-textual feature transformer, a module designed to transform visual features into a quasi-textual space closer to the textual domain via a c
    
[^2]: ParFormer：具有并行局部全局标记混合器和卷积注意力补丁嵌入的视觉Transformer基线

    ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding

    [https://arxiv.org/abs/2403.15004](https://arxiv.org/abs/2403.15004)

    ParFormer提出了并行局部全局标记混合器和卷积注意力补丁嵌入，优化了特征提取能力，在图像分类和对象识别等任务中表现优于CNN和最先进的Transformer架构。

    

    本文提出了ParFormer作为一种增强型Transformer架构，允许将不同的标记混合器整合到单个阶段中，从而提高特征提取能力。同时整合本地和全局数据，实现对短程和长程空间关系的精确表示，而无需像平移窗口这样需要大量计算的方法。除了并行标记混合器编码器外，我们提供了卷积注意力补丁嵌入(CAPE)，作为标准补丁嵌入的增强，通过卷积注意力模块改进标记混合器提取。我们的全面评估表明，我们的ParFormer在图像分类和物体识别等多个复杂任务中优于基于CNN和最先进的基于Transformer的架构。所提出的CAPE已被证明有益于整体MetaFormer架构，即使使用Id。

    arXiv:2403.15004v1 Announce Type: cross  Abstract: This work presents ParFormer as an enhanced transformer architecture that allows the incorporation of different token mixers into a single stage, hence improving feature extraction capabilities. Integrating both local and global data allows for precise representation of short- and long-range spatial relationships without the need for computationally intensive methods such as shifting windows. Along with the parallel token mixer encoder, We offer the Convolutional Attention Patch Embedding (CAPE) as an enhancement of standard patch embedding to improve token mixer extraction with a convolutional attention module. Our comprehensive evaluation demonstrates that our ParFormer outperforms CNN-based and state-of-the-art transformer-based architectures in image classification and several complex tasks such as object recognition. The proposed CAPE has been demonstrated to benefit the overall MetaFormer architecture, even while utilizing the Id
    
[^3]: 我们距离智能视觉演绎推理还有多远？

    How Far Are We from Intelligent Visual Deductive Reasoning?

    [https://arxiv.org/abs/2403.04732](https://arxiv.org/abs/2403.04732)

    目前的视觉语言模型在文本推理方面表现出色，但在视觉演绎推理方面仍存在较大差距和盲点。

    

    最近，诸如GPT-4V之类的视觉语言模型（VLM）在各种视觉语言任务上取得了巨大进展。我们深入探讨了基于视觉的演绎推理，这是一个更复杂但不太被探索的领域，并发现了当前领先的VLM中以前未暴露的盲点。具体来说，我们利用瑞文渐进矩阵（RPM）来评估VLM在仅依靠视觉线索进行多跳关系和演绎推理的能力。我们对几种流行的VLM进行了全面评估，采用了标准策略，如上下文学习、自我一致性和思维链（CoT），在三个不同的数据集上进行了评估，包括Mensa智商测试、智商测试和RAVEN。结果表明，尽管LLM在基于文本的推理方面具有令人印象深刻的能力，但我们在视觉演绎推理方面仍有很大的差距。

    arXiv:2403.04732v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) such as GPT-4V have recently demonstrated incredible strides on diverse vision language tasks. We dig into vision-based deductive reasoning, a more sophisticated but less explored realm, and find previously unexposed blindspots in the current SOTA VLMs. Specifically, we leverage Raven's Progressive Matrices (RPMs), to assess VLMs' abilities to perform multi-hop relational and deductive reasoning relying solely on visual clues. We perform comprehensive evaluations of several popular VLMs employing standard strategies such as in-context learning, self-consistency, and Chain-of-thoughts (CoT) on three diverse datasets, including the Mensa IQ test, IntelligenceTest, and RAVEN. The results reveal that despite the impressive capabilities of LLMs in text-based reasoning, we are still far from achieving comparable proficiency in visual deductive reasoning. We found that certain standard strategies that are effective
    
[^4]: 千面灌木：利用深度学习从卫星图像中进行个体分割

    Shrub of a thousand faces: an individual segmentation from satellite images using deep learning

    [https://arxiv.org/abs/2401.17985](https://arxiv.org/abs/2401.17985)

    该研究提出了一种新方法，利用深度学习模型从卫星图像中个别描绘位于西班牙内华达山脉林线上的杜松灌木。研究通过整合遥感影像和实例分割模型，解决了检测和描绘自然对象复杂生长模式的挑战。

    

    监测长寿灌木（如常见的杜松）的分布和大小结构可以用于估计气候变化对高山和高纬度生态系统的长期影响。历史航空超高分辨率影像提供了一种回顾性工具，可以高精度监测灌木的生长和分布。目前，深度学习模型在检测和描绘具有明确定义形状的对象的轮廓方面取得了令人印象深刻的结果。然而，将这些模型适应于检测表达复杂生长模式的自然对象（例如杜松）仍然是一项具有挑战性的任务。本研究提出了一种新方法，利用遥感RGB影像与基于Mask R-CNN的实例分割模型相结合，个别描绘西班牙内华达山脉林线上的杜松灌木。在这项研究中，我们提出了一种新的数据构造设计，其中使用的是光解译数据（PI）和实地调查数据（FW）分别进行操作。

    Monitoring the distribution and size structure of long-living shrubs, such as Juniperus communis, can be used to estimate the long-term effects of climate change on high-mountain and high latitude ecosystems. Historical aerial very-high resolution imagery offers a retrospective tool to monitor shrub growth and distribution at high precision. Currently, deep learning models provide impressive results for detecting and delineating the contour of objects with defined shapes. However, adapting these models to detect natural objects that express complex growth patterns, such as junipers, is still a challenging task.   This research presents a novel approach that leverages remotely sensed RGB imagery in conjunction with Mask R-CNN-based instance segmentation models to individually delineate Juniperus shrubs above the treeline in Sierra Nevada (Spain). In this study, we propose a new data construction design that consists in using photo interpreted (PI) and field work (FW) data to respectivel
    
[^5]: 通过多样化合成和扩散模型减轻偏见

    Mitigating Biases with Diverse Ensembles and Diffusion Models

    [https://arxiv.org/abs/2311.16176](https://arxiv.org/abs/2311.16176)

    通过利用扩散概率模型（DPMs）生成新特征组合的图像，可以在集成模型中增加模型多样性，并减轻捷径偏见，而无需额外监督信号。

    

    数据中的虚假相关性，即多个线索可以预测目标标签，常常导致一种称为捷径偏见的现象，即模型依赖于错误的、易学的线索，而忽略可靠的线索。在这项工作中，我们提出了一种利用扩散概率模型（DPMs）的集成多样化框架，用于减轻捷径偏见。我们展示了在特定的训练间隔中，DPMs可以生成具有新特征组合的图像，即使在显示相关输入特征的样本上进行训练。我们利用这一关键属性通过集成不一致性生成合成反事实来增加模型的多样性。我们展示了DPM引导的多样化足以消除对主要捷径线索的依赖，无需额外的监督信号。我们进一步在几个多样化目标上在实证上量化其有效性，并最终展示了改进的泛化性能。

    arXiv:2311.16176v2 Announce Type: replace-cross  Abstract: Spurious correlations in the data, where multiple cues are predictive of the target labels, often lead to a phenomenon known as shortcut bias, where a model relies on erroneous, easy-to-learn cues while ignoring reliable ones. In this work, we propose an ensemble diversification framework exploiting Diffusion Probabilistic Models (DPMs) for shortcut bias mitigation. We show that at particular training intervals, DPMs can generate images with novel feature combinations, even when trained on samples displaying correlated input features. We leverage this crucial property to generate synthetic counterfactuals to increase model diversity via ensemble disagreement. We show that DPM-guided diversification is sufficient to remove dependence on primary shortcut cues, without a need for additional supervised signals. We further empirically quantify its efficacy on several diversification objectives, and finally show improved generalizati
    
[^6]: 使用线性加法注意力Transformer的高效生成对抗网络

    Efficient generative adversarial networks using linear additive-attention Transformers. (arXiv:2401.09596v1 [cs.CV])

    [http://arxiv.org/abs/2401.09596](http://arxiv.org/abs/2401.09596)

    这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。

    

    尽管像扩散模型（DMs）和生成对抗网络（GANs）等深度生成模型在图像生成方面的能力近年来得到了显著提高，但是它们的成功很大程度上归功于计算复杂的架构。这限制了它们在研究实验室和资源充足的公司中的采用和使用，同时也极大地增加了训练、微调和推理的碳足迹。在这项工作中，我们提出了LadaGAN，这是一个高效的生成对抗网络，它建立在一种名为Ladaformer的新型Transformer块上。该块的主要组成部分是一个线性加法注意机制，它每个头部计算一个注意向量，而不是二次的点积注意力。我们在生成器和判别器中都采用了Ladaformer，这降低了计算复杂度，并克服了Transformer GAN经常出现的训练不稳定性。LadaGAN一直表现优于现有的GANs。

    Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms exist
    

