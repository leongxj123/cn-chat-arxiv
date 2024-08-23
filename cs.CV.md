# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue](https://arxiv.org/abs/2403.16276) | 介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。 |
| [^3] | [PhD: A Prompted Visual Hallucination Evaluation Dataset](https://arxiv.org/abs/2403.11116) | 本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。 |
| [^4] | [DiffuMatting: Synthesizing Arbitrary Objects with Matting-level Annotation](https://arxiv.org/abs/2403.06168) | 提出了DiffuMatting方法，通过扩散技术实现了“抠图任何物体”的能力，可以生成高度准确的注释，同时兼容社区LoRAs或各种条件控制方法。 |
| [^5] | [Efficient generative adversarial networks using linear additive-attention Transformers.](http://arxiv.org/abs/2401.09596) | 这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。 |
| [^6] | [Kosmos-2.5: A Multimodal Literate Model.](http://arxiv.org/abs/2309.11419) | Kosmos-2.5是一个多模态文学模型，能够在机器阅读文本密集型图像方面表现出色，并能够生成具有空间感的文本块和结构化文本输出。该模型具有通用性，可以适应不同提示下任何文本密集型图像理解任务，并为未来的扩展提供了方向。 |
| [^7] | [Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery.](http://arxiv.org/abs/2303.15975) | 本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。 |
| [^8] | [Predicting the Next Action by Modeling the Abstract Goal.](http://arxiv.org/abs/2209.05044) | 这篇论文提出了一种可以模型化抽象目标，以降低行动预测中不确定性的行动预测模型。使用视觉表征来描述动作和目标信息，并设计抽象目标为一个分布。该模型可在Epic-Kitchen数据集上实现最先进性能。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: AVicuna：具有交错器和上下文边界对齐的音频-视觉LLM用于时间指代对话

    AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue

    [https://arxiv.org/abs/2403.16276](https://arxiv.org/abs/2403.16276)

    介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。

    

    在日常交流中，人类经常使用语音和手势来指代特定区域或对象，这个过程称为指代对话（RD）。尽管先前的研究已经通过大型语言模型（LLMs）或大型多模型模型（LMMs）在静态环境中调查了RD，但在音频-视觉媒体中探索时间指代对话（TRD）仍然有限。两个主要挑战阻碍了这一领域的进展：（1）缺乏具有精确时间注释的全面未修剪音频-视觉视频数据集，以及（2）需要有效整合复杂的时间听觉和视觉线索的方法。为了解决这些挑战，我们引入了一个新的框架，生成PU-VALOR，这是一个包含超过114,000个未修剪视频的广泛音频-视觉数据集，并介绍了AVicuna，具有音频-视觉令牌交错器（AVTI），确保了时间对齐。

    arXiv:2403.16276v1 Announce Type: cross  Abstract: In everyday communication, humans frequently use speech and gestures to refer to specific areas or objects, a process known as Referential Dialogue (RD). While prior studies have investigated RD through Large Language Models (LLMs) or Large Multimodal Models (LMMs) in static contexts, the exploration of Temporal Referential Dialogue (TRD) within audio-visual media remains limited. Two primary challenges hinder progress in this field: (1) the absence of comprehensive, untrimmed audio-visual video datasets with precise temporal annotations, and (2) the need for methods to integrate complex temporal auditory and visual cues effectively. To address these challenges, we introduce a novel framework to generate PU-VALOR, an extensive audio-visual dataset comprising over 114,000 untrimmed videos with accurate temporal demarcations. We also present AVicuna, featuring an Audio-Visual Tokens Interleaver (AVTI) that ensures the temporal alignment 
    
[^3]: 博士论文：一个提示的视觉幻觉评估数据集

    PhD: A Prompted Visual Hallucination Evaluation Dataset

    [https://arxiv.org/abs/2403.11116](https://arxiv.org/abs/2403.11116)

    本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。

    

    大型语言模型（LLMs）的快速增长推动了大型视觉语言模型（LVLMs）的发展。在LLMs中普遍存在的幻觉挑战也出现在LVLMs中。然而，大部分现有研究主要集中在LVLM中的对象幻觉上，忽略了LVLM幻觉的多样化类型。本研究深入探讨了固有视觉语言幻觉（IVL-Hallu）问题，对导致幻觉的不同类型的IVL-Hallu进行了彻底分析。具体来说，我们提出了几个新颖的IVL-Hallu任务，并将它们分为四种类型：（a）对象幻觉，由于对象的误识别而产生，（b）属性幻觉，由于属性的误识别而引起，（c）多模态冲突幻觉，源自文本和视觉信息之间的矛盾，以及（d）反常识幻觉，由于对立之间的矛盾。

    arXiv:2403.11116v1 Announce Type: cross  Abstract: The rapid growth of Large Language Models (LLMs) has driven the development of Large Vision-Language Models (LVLMs). The challenge of hallucination, prevalent in LLMs, also emerges in LVLMs. However, most existing efforts mainly focus on object hallucination in LVLM, ignoring diverse types of LVLM hallucinations. In this study, we delve into the Intrinsic Vision-Language Hallucination (IVL-Hallu) issue, thoroughly analyzing different types of IVL-Hallu on their causes and reflections. Specifically, we propose several novel IVL-Hallu tasks and categorize them into four types: (a) object hallucination, which arises from the misidentification of objects, (b) attribute hallucination, which is caused by the misidentification of attributes, (c) multi-modal conflicting hallucination, which derives from the contradictions between textual and visual information, and (d) counter-common-sense hallucination, which owes to the contradictions betwee
    
[^4]: DiffuMatting：使用Matting级别标注合成任意对象

    DiffuMatting: Synthesizing Arbitrary Objects with Matting-level Annotation

    [https://arxiv.org/abs/2403.06168](https://arxiv.org/abs/2403.06168)

    提出了DiffuMatting方法，通过扩散技术实现了“抠图任何物体”的能力，可以生成高度准确的注释，同时兼容社区LoRAs或各种条件控制方法。

    

    由于获取高度准确或抠图注释的困难和劳动密集性，公开可用的高度准确标签数量有限。为了解决这一挑战，我们提出了一种DiffuMatting，它继承了扩散的强大生成能力，并赋予了“抠图任何物体”的能力。我们的DiffuMatting可以：1）作为一个具有高准确度注释的任意抠图工厂；2）与社区LoRAs或各种条件控制方法兼容，以实现社区友好的艺术设计和可控生成。具体地，受绿幕抠像的启发，我们旨在教授扩散模型在固定的绿幕画布上绘画。为此，收集了一个大规模的绿幕数据集（Green100K）作为DiffuMatting的训练数据集。其次，提出了一个绿色背景控制损失，以保持画布为纯绿色以进行区分。

    arXiv:2403.06168v1 Announce Type: cross  Abstract: Due to the difficulty and labor-consuming nature of getting highly accurate or matting annotations, there only exists a limited amount of highly accurate labels available to the public. To tackle this challenge, we propose a DiffuMatting which inherits the strong Everything generation ability of diffusion and endows the power of "matting anything". Our DiffuMatting can 1). act as an anything matting factory with high accurate annotations 2). be well-compatible with community LoRAs or various conditional control approaches to achieve the community-friendly art design and controllable generation. Specifically, inspired by green-screen-matting, we aim to teach the diffusion model to paint on a fixed green screen canvas. To this end, a large-scale greenscreen dataset (Green100K) is collected as a training dataset for DiffuMatting. Secondly, a green background control loss is proposed to keep the drawing board as a pure green color to disti
    
[^5]: 使用线性加法注意力Transformer的高效生成对抗网络

    Efficient generative adversarial networks using linear additive-attention Transformers. (arXiv:2401.09596v1 [cs.CV])

    [http://arxiv.org/abs/2401.09596](http://arxiv.org/abs/2401.09596)

    这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。

    

    尽管像扩散模型（DMs）和生成对抗网络（GANs）等深度生成模型在图像生成方面的能力近年来得到了显著提高，但是它们的成功很大程度上归功于计算复杂的架构。这限制了它们在研究实验室和资源充足的公司中的采用和使用，同时也极大地增加了训练、微调和推理的碳足迹。在这项工作中，我们提出了LadaGAN，这是一个高效的生成对抗网络，它建立在一种名为Ladaformer的新型Transformer块上。该块的主要组成部分是一个线性加法注意机制，它每个头部计算一个注意向量，而不是二次的点积注意力。我们在生成器和判别器中都采用了Ladaformer，这降低了计算复杂度，并克服了Transformer GAN经常出现的训练不稳定性。LadaGAN一直表现优于现有的GANs。

    Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms exist
    
[^6]: Kosmos-2.5: 一个多模态文学模型

    Kosmos-2.5: A Multimodal Literate Model. (arXiv:2309.11419v1 [cs.CL])

    [http://arxiv.org/abs/2309.11419](http://arxiv.org/abs/2309.11419)

    Kosmos-2.5是一个多模态文学模型，能够在机器阅读文本密集型图像方面表现出色，并能够生成具有空间感的文本块和结构化文本输出。该模型具有通用性，可以适应不同提示下任何文本密集型图像理解任务，并为未来的扩展提供了方向。

    

    我们介绍了Kosmos-2.5，一个用于对文本密集型图像进行机器阅读的多模态文学模型。Kosmos-2.5在两个不同但相互合作的转录任务中表现出色：(1) 生成具有空间感的文本块，其中每个文本块都被赋予其在图像中的空间坐标，以及(2) 生成以Markdown格式捕捉样式和结构的结构化文本输出。这种统一的多模态文学能力是通过共享的Transformer架构、任务特定的提示和灵活的文本表示实现的。我们在端到端的文档级文本识别和图像到Markdown文本生成上评估了Kosmos-2.5。此外，该模型可以通过监督微调轻松适应具有不同提示的任何文本密集型图像理解任务，使其成为涉及文本丰富图像的实际应用的通用工具。这项工作还为未来的扩展铺平了道路。

    We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling o
    
[^7]: 大规模预训练模型在增量式新类别发现中具有出乎意料的强大表现。

    Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery. (arXiv:2303.15975v1 [cs.CV])

    [http://arxiv.org/abs/2303.15975](http://arxiv.org/abs/2303.15975)

    本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。

    

    在生命长学习者中，从未标记的数据中连续地发现新概念是一个重要的期望。在文献中，这类问题在非常受限的情况下得到了部分解决，其中要么为发现新概念提供有标号的数据（例如 NCD），要么学习在有限数量的增量步骤中发生（例如类 iNCD）。在这项工作中，我们挑战现状，提出了一种更具挑战性和实用性的学习范式，称为 MSc-iNCD，其中学习连续而无人监督，并利用大规模预训练模型的丰富先验知识。为此，我们提出了简单的基线，不仅在较长的学习情境下具有弹性，而且与复杂的最先进方法相比，表现出出乎意料的强大实力。我们在多个基准测试中进行了广泛的实证评估，并展示了我们提出的基线的有效性，大大提升了基准要求。

    Discovering novel concepts from unlabelled data and in a continuous manner is an important desideratum of lifelong learners. In the literature such problems have been partially addressed under very restricted settings, where either access to labelled data is provided for discovering novel concepts (e.g., NCD) or learning occurs for a limited number of incremental steps (e.g., class-iNCD). In this work we challenge the status quo and propose a more challenging and practical learning paradigm called MSc-iNCD, where learning occurs continuously and unsupervisedly, while exploiting the rich priors from large-scale pre-trained models. To this end, we propose simple baselines that are not only resilient under longer learning scenarios, but are surprisingly strong when compared with sophisticated state-of-the-art methods. We conduct extensive empirical evaluation on a multitude of benchmarks and show the effectiveness of our proposed baselines, which significantly raises the bar.
    
[^8]: 建模抽象目标预测下一步动作

    Predicting the Next Action by Modeling the Abstract Goal. (arXiv:2209.05044v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.05044](http://arxiv.org/abs/2209.05044)

    这篇论文提出了一种可以模型化抽象目标，以降低行动预测中不确定性的行动预测模型。使用视觉表征来描述动作和目标信息，并设计抽象目标为一个分布。该模型可在Epic-Kitchen数据集上实现最先进性能。

    

    预测人类动作的问题具有固有的不确定性，但是，如果我们有关于动作实现目标的感知，可以降低这种不确定性。本文提出了一种行动预测模型，利用目标信息来减少未来预测中的不确定性。通过视觉表征，我们描述了动作和目标的信息。通过此方法，我们得出了一个称为抽象目标的新概念，其取决于观察到的视觉特征序列，用于行动预测。我们将抽象目标设计为一个分布，其参数是使用变分递归网络估计的。我们对下一个动作进行多次采样，并引入目标一致性度量来确定从抽象目标得出的最佳候选动作。我们的方法在极具挑战性的Epic-Kitchen数据集上取得了令人印象深刻的结果，并实现了最先进的性能。

    The problem of anticipating human actions is an inherently uncertain one. However, we can reduce this uncertainty if we have a sense of the goal that the actor is trying to achieve. Here, we present an action anticipation model that leverages goal information for the purpose of reducing the uncertainty in future predictions. Since we do not possess goal information or the observed actions during inference, we resort to visual representation to encapsulate information about both actions and goals. Through this, we derive a novel concept called abstract goal which is conditioned on observed sequences of visual features for action anticipation. We design the abstract goal as a distribution whose parameters are estimated using a variational recurrent network. We sample multiple candidates for the next action and introduce a goal consistency measure to determine the best candidate that follows from the abstract goal. Our method obtains impressive results on the very challenging Epic-Kitchen
    

