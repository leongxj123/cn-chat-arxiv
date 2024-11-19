# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SugarcaneNet2024: An Optimized Weighted Average Ensemble Approach of LASSO Regularized Pre-trained Models for Sugarcane Disease Classification](https://arxiv.org/abs/2403.18870) | SugarcaneNet2024是通过优化加权平均集成LASSO正则化的预训练模型，在甘蔗病害分类中表现出色，具有快速准确的检测能力。 |
| [^2] | [Brain Stroke Segmentation Using Deep Learning Models: A Comparative Study](https://arxiv.org/abs/2403.17177) | 本研究通过比较深度学习模型在脑卒中分割上的表现，探讨了是否需要高级别设计来获得最佳结果。 |
| [^3] | [PhD: A Prompted Visual Hallucination Evaluation Dataset](https://arxiv.org/abs/2403.11116) | 本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。 |
| [^4] | [Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume](https://arxiv.org/abs/2403.05100) | 提出新指标对抗超体积来全面评估深度学习模型在多种扰动强度下的鲁棒性，并采用新型训练算法来提高对抗鲁棒性。 |
| [^5] | [Pooling Image Datasets With Multiple Covariate Shift and Imbalance](https://arxiv.org/abs/2403.02598) | 本文从范畴论的角度提供了一个简单而有效的解决方案，完全避免了复杂的多阶段训练流程。 |
| [^6] | [Improving Contextual Congruence Across Modalities for Effective Multimodal Marketing using Knowledge-infused Learning](https://arxiv.org/abs/2402.03607) | 本研究提出了一种将常识知识图谱与大型视觉语言模型相结合的框架，用于改进预测多模态营销活动效果的性能。该方法能够提供早期检测可能具有说服力的多模态活动并评估和增强营销理论的能力。 |
| [^7] | [Efficient generative adversarial networks using linear additive-attention Transformers.](http://arxiv.org/abs/2401.09596) | 这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。 |
| [^8] | [A Comprehensive Study of Knowledge Editing for Large Language Models.](http://arxiv.org/abs/2401.01286) | 本研究全面研究了大型语言模型的知识编辑，旨在有效修改模型的行为，同时保持整体性能。 |
| [^9] | [A Scalable Training Strategy for Blind Multi-Distribution Noise Removal.](http://arxiv.org/abs/2310.20064) | 提出了一种使用自适应采样/主动学习策略来训练去噪网络的方法，解决了通用去噪网络在不同噪声分布下表现差的问题。 |
| [^10] | [Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog.](http://arxiv.org/abs/2310.07259) | 本文提出了一种迭代跟踪和推理策略，结合文本编码器和视觉编码器以生成准确的响应，解决了视频对话中逐步理解对话历史和吸收视频信息的挑战。 |
| [^11] | [Unmasking Parkinson's Disease with Smile: An AI-enabled Screening Framework.](http://arxiv.org/abs/2308.02588) | 本研究使用微表情视频数据集开发了一种基于人工智能的帕金森病筛查框架，通过分析微笑视频中的特征，实现了89.7%的准确性和89.3%的AUROC值，同时在人群子组上没有检测到偏见。 |
| [^12] | [A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning.](http://arxiv.org/abs/2307.09218) | 遗忘是深度学习中普遍存在的现象，不仅限于连续学习领域。解决遗忘问题面临多个挑战，包括平衡保留旧任务知识与快速学习新任务的挑战，管理任务干扰与冲突目标的挑战，以及防止隐私泄露等。遗忘不总是有害的，可以在某些情况下是有益且可取的，特别是在隐私保护场景中。 |

# 详细

[^1]: SugarcaneNet2024: LASSO正则化的预训练模型的优化加权平均集成方法用于甘蔗病害分类

    SugarcaneNet2024: An Optimized Weighted Average Ensemble Approach of LASSO Regularized Pre-trained Models for Sugarcane Disease Classification

    [https://arxiv.org/abs/2403.18870](https://arxiv.org/abs/2403.18870)

    SugarcaneNet2024是通过优化加权平均集成LASSO正则化的预训练模型，在甘蔗病害分类中表现出色，具有快速准确的检测能力。

    

    甘蔗作为世界糖业的关键作物，容易受多种病害侵害，这些病害对其产量和质量都有重大负面影响。为了有效管理和实施预防措施，必须及时准确地检测病害。本研究提出了一种名为SugarcaneNet2024的独特模型，通过叶片图像处理，能够优于先前方法自动快速检测甘蔗病害。我们提出的模型汇总了七个定制的、经过LASSO正则化的预训练模型的优化加权平均集成，特别是InceptionV3、InceptionResNetV2、DenseNet201、DenseNet169、Xception和ResNet152V2。最初，我们在这些预训练模型底部添加了三层更密集层，具有0.0001的LASSO正则化，三个30%的dropout层和三个启用renorm的批量归一化，以提高性能。

    arXiv:2403.18870v1 Announce Type: cross  Abstract: Sugarcane, a key crop for the world's sugar industry, is prone to several diseases that have a substantial negative influence on both its yield and quality. To effectively manage and implement preventative initiatives, diseases must be detected promptly and accurately. In this study, we present a unique model called sugarcaneNet2024 that outperforms previous methods for automatically and quickly detecting sugarcane disease through leaf image processing. Our proposed model consolidates an optimized weighted average ensemble of seven customized and LASSO-regularized pre-trained models, particularly InceptionV3, InceptionResNetV2, DenseNet201, DenseNet169, Xception, and ResNet152V2. Initially, we added three more dense layers with 0.0001 LASSO regularization, three 30% dropout layers, and three batch normalizations with renorm enabled at the bottom of these pre-trained models to improve the performance. The accuracy of sugarcane leaf dise
    
[^2]: 使用深度学习模型进行脑卒中分割：一项比较研究

    Brain Stroke Segmentation Using Deep Learning Models: A Comparative Study

    [https://arxiv.org/abs/2403.17177](https://arxiv.org/abs/2403.17177)

    本研究通过比较深度学习模型在脑卒中分割上的表现，探讨了是否需要高级别设计来获得最佳结果。

    

    脑卒中分割在脑卒中患者的诊断和治疗中发挥着关键作用，通过提供受影响脑区域的空间信息和受损程度。准确分割脑卒中病变是一项具有挑战性的任务，因为传统的手工技术耗时且容易出错。最近，先进的深度模型已被引入用于一般医学图像分割，展示出在特定数据集上评估时超越许多最先进网络的有前景结果。随着视觉Transformer的出现，已经基于它们引入了几种模型，而其他一些则旨在设计基于传统卷积层来提取像Transformer这样的长程依赖的更好模块。是否对所有分割案例都需要这样高级别的设计来实现最佳结果的问题尚未得到解答。在这项研究中，我们选择了四种类型的深度学习模型

    arXiv:2403.17177v1 Announce Type: cross  Abstract: Stroke segmentation plays a crucial role in the diagnosis and treatment of stroke patients by providing spatial information about affected brain regions and the extent of damage. Segmenting stroke lesions accurately is a challenging task, given that conventional manual techniques are time consuming and prone to errors. Recently, advanced deep models have been introduced for general medical image segmentation, demonstrating promising results that surpass many state of the art networks when evaluated on specific datasets. With the advent of the vision Transformers, several models have been introduced based on them, while others have aimed to design better modules based on traditional convolutional layers to extract long-range dependencies like Transformers. The question of whether such high-level designs are necessary for all segmentation cases to achieve the best results remains unanswered. In this study, we selected four types of deep 
    
[^3]: 博士论文：一个提示的视觉幻觉评估数据集

    PhD: A Prompted Visual Hallucination Evaluation Dataset

    [https://arxiv.org/abs/2403.11116](https://arxiv.org/abs/2403.11116)

    本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。

    

    大型语言模型（LLMs）的快速增长推动了大型视觉语言模型（LVLMs）的发展。在LLMs中普遍存在的幻觉挑战也出现在LVLMs中。然而，大部分现有研究主要集中在LVLM中的对象幻觉上，忽略了LVLM幻觉的多样化类型。本研究深入探讨了固有视觉语言幻觉（IVL-Hallu）问题，对导致幻觉的不同类型的IVL-Hallu进行了彻底分析。具体来说，我们提出了几个新颖的IVL-Hallu任务，并将它们分为四种类型：（a）对象幻觉，由于对象的误识别而产生，（b）属性幻觉，由于属性的误识别而引起，（c）多模态冲突幻觉，源自文本和视觉信息之间的矛盾，以及（d）反常识幻觉，由于对立之间的矛盾。

    arXiv:2403.11116v1 Announce Type: cross  Abstract: The rapid growth of Large Language Models (LLMs) has driven the development of Large Vision-Language Models (LVLMs). The challenge of hallucination, prevalent in LLMs, also emerges in LVLMs. However, most existing efforts mainly focus on object hallucination in LVLM, ignoring diverse types of LVLM hallucinations. In this study, we delve into the Intrinsic Vision-Language Hallucination (IVL-Hallu) issue, thoroughly analyzing different types of IVL-Hallu on their causes and reflections. Specifically, we propose several novel IVL-Hallu tasks and categorize them into four types: (a) object hallucination, which arises from the misidentification of objects, (b) attribute hallucination, which is caused by the misidentification of attributes, (c) multi-modal conflicting hallucination, which derives from the contradictions between textual and visual information, and (d) counter-common-sense hallucination, which owes to the contradictions betwee
    
[^4]: 探索对抗界限：通过对抗超体积量化鲁棒性

    Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume

    [https://arxiv.org/abs/2403.05100](https://arxiv.org/abs/2403.05100)

    提出新指标对抗超体积来全面评估深度学习模型在多种扰动强度下的鲁棒性，并采用新型训练算法来提高对抗鲁棒性。

    

    在深度学习模型面临日益严重的对抗攻击威胁，特别是在安全关键领域，强调了对鲁棒深度学习系统的需求。传统的鲁棒性评估依赖于对抗准确性，该指标衡量模型在特定扰动强度下的性能。然而，这一单一指标并不能完全概括模型对不同程度扰动的整体韧性。为了填补这一空白，我们提出了一种新的指标，称为对抗超体积，从多目标优化的角度综合评估了深度学习模型在一系列扰动强度下的鲁棒性。该指标允许深入比较防御机制，并承认了较弱的防御策略所带来的鲁棒性改进。此外，我们采用了一种提高对抗鲁棒性均匀性的新型训练算法。

    arXiv:2403.05100v1 Announce Type: cross  Abstract: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly
    
[^5]: 具有多个协变量转移和不平衡的图像数据集聚合

    Pooling Image Datasets With Multiple Covariate Shift and Imbalance

    [https://arxiv.org/abs/2403.02598](https://arxiv.org/abs/2403.02598)

    本文从范畴论的角度提供了一个简单而有效的解决方案，完全避免了复杂的多阶段训练流程。

    

    许多学科中常见小样本大小，这需要跨多个机构汇总大致相似的数据集来研究图像与疾病结果之间的弱但相关关联。这些数据通常体现出协变量（即次要的非成像数据）的转移/不平衡。在标准统计分析中控制这些无用变量是常见的，但这些思想并不直接适用于参数过多的模型。因此，最近的工作表明，从不变表示学习中提供了一个有意义的起点，但目前的方法库仅限于一次考虑几个协变量的转移/不平衡。本文展示了如何从范畴论的角度看待这一问题，提供了一个简单而有效的解决方案，完全避免了原本需要复杂的多阶段训练流程。我们展示了该方法的效果。

    arXiv:2403.02598v1 Announce Type: new  Abstract: Small sample sizes are common in many disciplines, which necessitates pooling roughly similar datasets across multiple institutions to study weak but relevant associations between images and disease outcomes. Such data often manifest shift/imbalance in covariates (i.e., secondary non-imaging data). Controlling for such nuisance variables is common within standard statistical analysis, but the ideas do not directly apply to overparameterized models. Consequently, recent work has shown how strategies from invariant representation learning provides a meaningful starting point, but the current repertoire of methods is limited to accounting for shifts/imbalances in just a couple of covariates at a time. In this paper, we show how viewing this problem from the perspective of Category theory provides a simple and effective solution that completely avoids elaborate multi-stage training pipelines that would otherwise be needed. We show the effect
    
[^6]: 提高多模态营销的上下文一致性：知识基础学习的有效性

    Improving Contextual Congruence Across Modalities for Effective Multimodal Marketing using Knowledge-infused Learning

    [https://arxiv.org/abs/2402.03607](https://arxiv.org/abs/2402.03607)

    本研究提出了一种将常识知识图谱与大型视觉语言模型相结合的框架，用于改进预测多模态营销活动效果的性能。该方法能够提供早期检测可能具有说服力的多模态活动并评估和增强营销理论的能力。

    

    智能设备的普及使用户能够在线体验多模态信息。然而，大型语言模型（LLM）和视觉模型（LVM）仍然受到捕捉跨模态语义关系的整体意义的限制。缺乏明确的常识知识（例如，作为一个知识图谱），视觉语言模型（VLM）仅通过捕捉庞大的语料库中的高级模式来学习隐式表示，从而忽略了重要的上下文跨模态线索。在这项工作中，我们设计了一个框架，将显式的常识知识以知识图谱的形式与大型的VLM相结合，以提高下游任务的性能，即预测多模态营销活动的有效性。虽然营销应用提供了一个有说服力的指标来评估我们的方法，但我们的方法使得早期发现可能具有说服力的多模态活动成为可能，并评估和增强营销理论。

    The prevalence of smart devices with the ability to capture moments in multiple modalities has enabled users to experience multimodal information online. However, large Language (LLMs) and Vision models (LVMs) are still limited in capturing holistic meaning with cross-modal semantic relationships. Without explicit, common sense knowledge (e.g., as a knowledge graph), Visual Language Models (VLMs) only learn implicit representations by capturing high-level patterns in vast corpora, missing essential contextual cross-modal cues. In this work, we design a framework to couple explicit commonsense knowledge in the form of knowledge graphs with large VLMs to improve the performance of a downstream task, predicting the effectiveness of multi-modal marketing campaigns. While the marketing application provides a compelling metric for assessing our methods, our approach enables the early detection of likely persuasive multi-modal campaigns and the assessment and augmentation of marketing theory.
    
[^7]: 使用线性加法注意力Transformer的高效生成对抗网络

    Efficient generative adversarial networks using linear additive-attention Transformers. (arXiv:2401.09596v1 [cs.CV])

    [http://arxiv.org/abs/2401.09596](http://arxiv.org/abs/2401.09596)

    这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。

    

    尽管像扩散模型（DMs）和生成对抗网络（GANs）等深度生成模型在图像生成方面的能力近年来得到了显著提高，但是它们的成功很大程度上归功于计算复杂的架构。这限制了它们在研究实验室和资源充足的公司中的采用和使用，同时也极大地增加了训练、微调和推理的碳足迹。在这项工作中，我们提出了LadaGAN，这是一个高效的生成对抗网络，它建立在一种名为Ladaformer的新型Transformer块上。该块的主要组成部分是一个线性加法注意机制，它每个头部计算一个注意向量，而不是二次的点积注意力。我们在生成器和判别器中都采用了Ladaformer，这降低了计算复杂度，并克服了Transformer GAN经常出现的训练不稳定性。LadaGAN一直表现优于现有的GANs。

    Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms exist
    
[^8]: 大型语言模型的知识编辑全面研究

    A Comprehensive Study of Knowledge Editing for Large Language Models. (arXiv:2401.01286v1 [cs.CL])

    [http://arxiv.org/abs/2401.01286](http://arxiv.org/abs/2401.01286)

    本研究全面研究了大型语言模型的知识编辑，旨在有效修改模型的行为，同时保持整体性能。

    

    大型语言模型(LLM)在理解和生成与人类交流紧密相似的文本方面展现出了非凡的能力。然而，其主要限制在于训练过程中的显著计算需求，这是由于其广泛的参数化造成的。这一挑战在于世界的动态性，需要频繁更新LLM以修正过时的信息或集成新知识，从而确保其持续的相关性。许多应用需要在训练后进行持续的模型调整，以解决缺陷或不良行为。近年来，对于LLM的知识编辑技术的兴趣越来越高，在特定领域内有效地修改LLM的行为，同时保持整体性能在各种输入中的表现。本文首先定义了知识编辑的目标和挑战，然后综述了现有的知识编辑方法和技术，并讨论了其应用和未来发展的方向。

    Large Language Models (LLMs) have shown extraordinary capabilities in understanding and generating text that closely mirrors human communication. However, a primary limitation lies in the significant computational demands during training, arising from their extensive parameterization. This challenge is further intensified by the dynamic nature of the world, necessitating frequent updates to LLMs to correct outdated information or integrate new knowledge, thereby ensuring their continued relevance. Note that many applications demand continual model adjustments post-training to address deficiencies or undesirable behaviors. There is an increasing interest in efficient, lightweight methods for on-the-fly model modifications. To this end, recent years have seen a burgeoning in the techniques of knowledge editing for LLMs, which aim to efficiently modify LLMs' behaviors within specific domains while preserving overall performance across various inputs. In this paper, we first define the kno
    
[^9]: 一种可扩展的训练策略用于盲目的多分布噪声去除

    A Scalable Training Strategy for Blind Multi-Distribution Noise Removal. (arXiv:2310.20064v1 [cs.CV])

    [http://arxiv.org/abs/2310.20064](http://arxiv.org/abs/2310.20064)

    提出了一种使用自适应采样/主动学习策略来训练去噪网络的方法，解决了通用去噪网络在不同噪声分布下表现差的问题。

    

    尽管最近取得了一些进展，但是开发通用的去噪和去伪影网络仍然是一个尚未解决的问题：给定固定的网络权重，一个任务（例如去除泊松噪声）的专门化与另一个任务（例如去除斑点噪声）的性能之间存在天然的权衡。此外，由于维度的诅咒，训练这样的网络是具有挑战性的：随着规格空间的维度增加（即需要描述噪声分布所需的参数数量增加），需要训练的唯一规格数量呈指数增长。均匀采样这个空间会导致网络在非常具有挑战性的问题规格上表现良好，但在简单的问题规格上表现不佳，即使大误差也对总体均方误差的影响很小。本文提出了一种使用自适应采样/主动学习策略来训练去噪网络的方法。我们的工作改进了最近提出的一种方法。

    Despite recent advances, developing general-purpose universal denoising and artifact-removal networks remains largely an open problem: Given fixed network weights, one inherently trades-off specialization at one task (e.g.,~removing Poisson noise) for performance at another (e.g.,~removing speckle noise). In addition, training such a network is challenging due to the curse of dimensionality: As one increases the dimensions of the specification-space (i.e.,~the number of parameters needed to describe the noise distribution) the number of unique specifications one needs to train for grows exponentially. Uniformly sampling this space will result in a network that does well at very challenging problem specifications but poorly at easy problem specifications, where even large errors will have a small effect on the overall mean squared error.  In this work we propose training denoising networks using an adaptive-sampling/active-learning strategy. Our work improves upon a recently proposed un
    
[^10]: 揭示隐藏的联系：用于视频对话的迭代跟踪和推理

    Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog. (arXiv:2310.07259v1 [cs.CV])

    [http://arxiv.org/abs/2310.07259](http://arxiv.org/abs/2310.07259)

    本文提出了一种迭代跟踪和推理策略，结合文本编码器和视觉编码器以生成准确的响应，解决了视频对话中逐步理解对话历史和吸收视频信息的挑战。

    

    与传统的视觉问答相比，视频对话需要对对话历史和视频内容进行深入理解，以生成准确的响应。尽管现有的方法取得了令人称赞的进展，但它们常常面临逐步理解复杂的对话历史和吸收视频信息的挑战。为了弥补这一差距，我们提出了一种迭代跟踪和推理策略，将文本编码器、视觉编码器和生成器相结合。我们的文本编码器以路径跟踪和聚合机制为核心，能够从对话历史中获取重要的细微差别，以解释所提出的问题。同时，我们的视觉编码器利用迭代推理网络，精心设计以从视频中提取和强调关键视觉标记，增强对视觉理解的深度。最后，我们使用预训练的GPT-模型将这些丰富的信息综合起来。

    In contrast to conventional visual question answering, video-grounded dialog necessitates a profound understanding of both dialog history and video content for accurate response generation. Despite commendable strides made by existing methodologies, they often grapple with the challenges of incrementally understanding intricate dialog histories and assimilating video information. In response to this gap, we present an iterative tracking and reasoning strategy that amalgamates a textual encoder, a visual encoder, and a generator. At its core, our textual encoder is fortified with a path tracking and aggregation mechanism, adept at gleaning nuances from dialog history that are pivotal to deciphering the posed questions. Concurrently, our visual encoder harnesses an iterative reasoning network, meticulously crafted to distill and emphasize critical visual markers from videos, enhancing the depth of visual comprehension. Culminating this enriched information, we employ the pre-trained GPT-
    
[^11]: 用微笑揭示帕金森病：一种基于人工智能的筛查框架

    Unmasking Parkinson's Disease with Smile: An AI-enabled Screening Framework. (arXiv:2308.02588v1 [eess.IV])

    [http://arxiv.org/abs/2308.02588](http://arxiv.org/abs/2308.02588)

    本研究使用微表情视频数据集开发了一种基于人工智能的帕金森病筛查框架，通过分析微笑视频中的特征，实现了89.7%的准确性和89.3%的AUROC值，同时在人群子组上没有检测到偏见。

    

    鉴于目前缺乏可靠的生物标志物和有限的临床护理资源，帕金森病（PD）的诊断仍然具有挑战性。在本研究中，我们使用包含微表情的最大视频数据集进行PD筛查的分析。我们收集了来自1,059名独立参与者的3,871个视频，其中包括256名自报PD患者。这些录像来自不同来源，包括多个国家的参与者家中、一家诊所和一个美国的PD护理机构。通过利用面部标志和行动单位，我们提取了与PD的一个主要症状Hypomimia（面部表情减少）相关的特征。在这些特征上训练的一组AI模型在保留数据上实现了89.7%的准确性和89.3%的接收者操作特性曲线下面积（AUROC），并且在性别和种族等人群子组上无可检测的偏见。进一步的分析揭示，仅通过微笑视频中的特征就可以获得可比较的准确性和AUROC值。

    Parkinson's disease (PD) diagnosis remains challenging due to lacking a reliable biomarker and limited access to clinical care. In this study, we present an analysis of the largest video dataset containing micro-expressions to screen for PD. We collected 3,871 videos from 1,059 unique participants, including 256 self-reported PD patients. The recordings are from diverse sources encompassing participants' homes across multiple countries, a clinic, and a PD care facility in the US. Leveraging facial landmarks and action units, we extracted features relevant to Hypomimia, a prominent symptom of PD characterized by reduced facial expressions. An ensemble of AI models trained on these features achieved an accuracy of 89.7% and an Area Under the Receiver Operating Characteristic (AUROC) of 89.3% while being free from detectable bias across population subgroups based on sex and ethnicity on held-out data. Further analysis reveals that features from the smiling videos alone lead to comparable 
    
[^12]: 深度学习中遗忘现象的全面调查：超越连续学习

    A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning. (arXiv:2307.09218v1 [cs.LG])

    [http://arxiv.org/abs/2307.09218](http://arxiv.org/abs/2307.09218)

    遗忘是深度学习中普遍存在的现象，不仅限于连续学习领域。解决遗忘问题面临多个挑战，包括平衡保留旧任务知识与快速学习新任务的挑战，管理任务干扰与冲突目标的挑战，以及防止隐私泄露等。遗忘不总是有害的，可以在某些情况下是有益且可取的，特别是在隐私保护场景中。

    

    遗忘指的是先前获取的信息或知识的丧失或恶化。尽管现有的关于遗忘的调查主要集中在连续学习方面，但在深度学习中，遗忘是一种普遍现象，可以在各种其他研究领域中观察到。遗忘在研究领域中表现出来，例如由于生成器漂移而在生成模型领域中表现出来，以及由于客户端之间存在异构数据分布而在联邦学习中表现出来。解决遗忘问题涉及到几个挑战，包括在快速学习新任务的同时平衡保留旧任务知识，管理任务干扰与冲突目标，以及防止隐私泄露等。此外，大多数现有的连续学习调查都默认认为遗忘总是有害的。相反，我们的调查认为遗忘是一把双刃剑，在某些情况下可以是有益且可取的，例如隐私保护场景。通过在更广泛的背景下探讨遗忘现象，

    Forgetting refers to the loss or deterioration of previously acquired information or knowledge. While the existing surveys on forgetting have primarily focused on continual learning, forgetting is a prevalent phenomenon observed in various other research domains within deep learning. Forgetting manifests in research fields such as generative models due to generator shifts, and federated learning due to heterogeneous data distributions across clients. Addressing forgetting encompasses several challenges, including balancing the retention of old task knowledge with fast learning of new tasks, managing task interference with conflicting goals, and preventing privacy leakage, etc. Moreover, most existing surveys on continual learning implicitly assume that forgetting is always harmful. In contrast, our survey argues that forgetting is a double-edged sword and can be beneficial and desirable in certain cases, such as privacy-preserving scenarios. By exploring forgetting in a broader context
    

