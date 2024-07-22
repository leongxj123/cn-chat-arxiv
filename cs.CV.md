# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection](https://arxiv.org/abs/2403.14270) | 提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。 |
| [^2] | [Debiasing surgeon: fantastic weights and how to find them](https://arxiv.org/abs/2403.14200) | 证明了在深度学习模型中存在一些无偏子网络，可以在不需要依赖算法偏见的情况下被提取出来，并且这种特定架构无法学习任何特定的偏见。 |
| [^3] | [VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](https://arxiv.org/abs/2403.12034) | 利用预训练的视频扩散模型，本文提出了一个可生成大规模3D数据集的VFusion3D模型。 |
| [^4] | [Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning](https://arxiv.org/abs/2403.10107) | 通过多个大型预训练语言模型的合作推理，本研究提出了V-HOI Multi-LLMs Collaborated Reasoning（V-HOI MLCR）框架，用于增强当前V-HOI检测模型的性能。 |
| [^5] | [Towards Scene Graph Anticipation](https://arxiv.org/abs/2403.04899) | 提出了场景图预测（SGA）任务，并引入一个新的方法SceneSayer，通过使用神经ODE和神经SDE的概念，结合对象-centric的关系表示，实现对象之间未来关系的预测。 |
| [^6] | [Layerwise complexity-matched learning yields an improved model of cortical area V2](https://arxiv.org/abs/2312.11436) | 通过分层复杂度匹配学习，我们开发了一种自下而上的自监督训练方法，最大化了特征相似性同时在不同位置的补丁上解除特征相关性。 |
| [^7] | [Fast Registration of Photorealistic Avatars for VR Facial Animation.](http://arxiv.org/abs/2401.11002) | 本论文针对虚拟现实头像注册和面部动画问题，发现头像和头显相机图像之间的领域差距是主要难点，并提出了一个系统设计来解决这个问题。 |
| [^8] | [Adversarial Examples in the Physical World: A Survey.](http://arxiv.org/abs/2311.01473) | 本综述系统地研究了物理世界中的对抗样本（PAEs）的特点，并提出了基于其特征的全面分析和分类框架，涵盖了100多个研究，以填补对PAEs独特特征的现有研究不足。 |
| [^9] | [Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation.](http://arxiv.org/abs/2310.07506) | 本文提出了一种利用分层特征共享的数据参数化架构（HMN），旨在更高效地压缩数据。通过将数据存储在三层结构中，HMN能够捕捉到数据集级别、类别级别和样本级别的特征。 |
| [^10] | [Data Augmentation through Pseudolabels in Automatic Region Based Coronary Artery Segmentation for Disease Diagnosis.](http://arxiv.org/abs/2310.05990) | 这项研究引入了伪标签作为数据增强技术，通过改善基准Yolo模型的性能，提高了冠状动脉分割的效果。 |
| [^11] | [Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection.](http://arxiv.org/abs/2310.00161) | 这项研究提出了一种面向检测的图像-文本预训练方法，旨在弥合图像级预训练和开放词汇目标检测之间的差距。通过检测器架构和对比损失，该方法能够从噪声图像-文本对中学习到新出现的物体-语义线索，并提出了一种平移窗口学习方法来改进主干网络的表示。在LVIS开放词汇检测基准上，该方法取得了显著优于其他方法的40.4的掩码AP$_r$结果。 |
| [^12] | [Cross-Validation Is All You Need: A Statistical Approach To Label Noise Estimation.](http://arxiv.org/abs/2306.13990) | 本论文提出了一种重复交叉验证(Repeated Cross-Validation)方法，通过构建噪声直方图并提出三种基于该直方图的方法来检测标签噪声并清理数据，解决了结果预测分析中的数据清洗问题。 |

# 详细

[^1]: 场景图ViT：端到端的开放词汇视觉关系检测

    Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection

    [https://arxiv.org/abs/2403.14270](https://arxiv.org/abs/2403.14270)

    提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。

    

    视觉关系检测旨在识别图像中的对象及其关系。以往的方法通过在现有目标检测架构中添加单独的关系模块或解码器来处理此任务。这种分离增加了复杂性，阻碍了端到端训练，限制了性能。我们提出了一种简单且高效的无解码器架构，用于开放词汇的视觉关系检测。我们的模型由基于Transformer的图像编码器组成，将对象表示为标记，并隐含地建模它们的关系。为了提取关系信息，我们引入了一个注意力机制，选择可能形成关系的对象对。我们提供了一个单阶段的训练方法，可以在混合对象和关系检测数据上训练此模型。我们的方法在Visual Genome和大词汇GQA基准测试上实现了最先进的关系检测性能，可实现实时性。

    arXiv:2403.14270v1 Announce Type: cross  Abstract: Visual relationship detection aims to identify objects and their relationships in images. Prior methods approach this task by adding separate relationship modules or decoders to existing object detection architectures. This separation increases complexity and hinders end-to-end training, which limits performance. We propose a simple and highly efficient decoder-free architecture for open-vocabulary visual relationship detection. Our model consists of a Transformer-based image encoder that represents objects as tokens and models their relationships implicitly. To extract relationship information, we introduce an attention mechanism that selects object pairs likely to form a relationship. We provide a single-stage recipe to train this model on a mixture of object and relationship detection data. Our approach achieves state-of-the-art relationship detection performance on Visual Genome and on the large-vocabulary GQA benchmark at real-tim
    
[^2]: 手术员去偏见：神奇的权重及如何找到它们

    Debiasing surgeon: fantastic weights and how to find them

    [https://arxiv.org/abs/2403.14200](https://arxiv.org/abs/2403.14200)

    证明了在深度学习模型中存在一些无偏子网络，可以在不需要依赖算法偏见的情况下被提取出来，并且这种特定架构无法学习任何特定的偏见。

    

    现今一个日益关注的现象是算法偏见的出现，它可能导致不公平的模型。在深度学习领域，已经提出了几种去偏见的方法，采用更或多或少复杂的方法来阻止这些模型大规模地使用这些偏见。然而，一个问题出现了：这种额外的复杂性真的有必要吗？一个普通训练的模型是否已经包含了一些可以独立使用的“无偏子网络”，并且可以提出一个解决方案而不依赖于算法偏见？在这项工作中，我们展示了这样的子网络通常存在，并且可以从一个普通训练的模型中提取出来，而无需额外的训练。我们进一步验证了这种特定的架构无法学习特定的偏见，表明在深度神经网络中有可能通过架构上的对策来解决偏见问题。

    arXiv:2403.14200v1 Announce Type: cross  Abstract: Nowadays an ever-growing concerning phenomenon, the emergence of algorithmic biases that can lead to unfair models, emerges. Several debiasing approaches have been proposed in the realm of deep learning, employing more or less sophisticated approaches to discourage these models from massively employing these biases. However, a question emerges: is this extra complexity really necessary? Is a vanilla-trained model already embodying some ``unbiased sub-networks'' that can be used in isolation and propose a solution without relying on the algorithmic biases? In this work, we show that such a sub-network typically exists, and can be extracted from a vanilla-trained model without requiring additional training. We further validate that such specific architecture is incapable of learning a specific bias, suggesting that there are possible architectural countermeasures to the problem of biases in deep neural networks.
    
[^3]: VFusion3D: 从视频扩散模型中学习可扩展的3D生成模型

    VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models

    [https://arxiv.org/abs/2403.12034](https://arxiv.org/abs/2403.12034)

    利用预训练的视频扩散模型，本文提出了一个可生成大规模3D数据集的VFusion3D模型。

    

    本文提出了一种新颖的范式，利用预训练的视频扩散模型构建可扩展的3D生成模型。构建基础3D生成模型的主要障碍是3D数据的有限可用性。与图像、文本或视频不同，3D数据不容易获取且难以获得，这导致与其他类型数据的数量相比存在显着的规模差异。为了解决这个问题，我们提出使用一个通过大量文本、图像和视频训练的视频扩散模型作为3D数据的知识源。通过微调解锁其多视角生成能力，我们生成一个大规模的合成多视角数据集来训练前馈3D生成模型。

    arXiv:2403.12034v1 Announce Type: cross  Abstract: This paper presents a novel paradigm for building scalable 3D generative models utilizing pre-trained video diffusion models. The primary obstacle in developing foundation 3D generative models is the limited availability of 3D data. Unlike images, texts, or videos, 3D data are not readily accessible and are difficult to acquire. This results in a significant disparity in scale compared to the vast quantities of other types of data. To address this issue, we propose using a video diffusion model, trained with extensive volumes of text, images, and videos, as a knowledge source for 3D data. By unlocking its multi-view generative capabilities through fine-tuning, we generate a large-scale synthetic multi-view dataset to train a feed-forward 3D generative model. The proposed model, VFusion3D, trained on nearly 3M synthetic multi-view data, can generate a 3D asset from a single image in seconds and achieves superior performance when compare
    
[^4]: 通过多个LLM合作推理提升人类中心动态场景理解

    Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning

    [https://arxiv.org/abs/2403.10107](https://arxiv.org/abs/2403.10107)

    通过多个大型预训练语言模型的合作推理，本研究提出了V-HOI Multi-LLMs Collaborated Reasoning（V-HOI MLCR）框架，用于增强当前V-HOI检测模型的性能。

    

    人类中心的动态场景理解在增强机器人和自主系统的能力中起着至关重要的作用，其中视频人-物交互（V-HOI）检测是语义场景理解中的关键任务，旨在全面理解视频中的HOI关系，以使移动机器人和自动驾驶系统的行为决策受益。虽然先前的V-HOI检测模型在特定数据集上取得了显著进展，但它们仍然缺乏像人类一样的通用推理能力，无法有效引导HOI关系。在本研究中，我们提出了V-HOI多LLM协同推理（V-HOI MLCR），这是一个新颖的框架，由一系列即插即用的模块组成，可以通过利用不同现成大型预训练语言模型（LLMs）的强大推理能力，促进当前V-HOI检测模型的性能。

    arXiv:2403.10107v1 Announce Type: cross  Abstract: Human-centered dynamic scene understanding plays a pivotal role in enhancing the capability of robotic and autonomous systems, in which Video-based Human-Object Interaction (V-HOI) detection is a crucial task in semantic scene understanding, aimed at comprehensively understanding HOI relationships within a video to benefit the behavioral decisions of mobile robots and autonomous driving systems. Although previous V-HOI detection models have made significant strides in accurate detection on specific datasets, they still lack the general reasoning ability like human beings to effectively induce HOI relationships. In this study, we propose V-HOI Multi-LLMs Collaborated Reasoning (V-HOI MLCR), a novel framework consisting of a series of plug-and-play modules that could facilitate the performance of current V-HOI detection models by leveraging the strong reasoning ability of different off-the-shelf pre-trained large language models (LLMs). 
    
[^5]: 朝向场景图预测

    Towards Scene Graph Anticipation

    [https://arxiv.org/abs/2403.04899](https://arxiv.org/abs/2403.04899)

    提出了场景图预测（SGA）任务，并引入一个新的方法SceneSayer，通过使用神经ODE和神经SDE的概念，结合对象-centric的关系表示，实现对象之间未来关系的预测。

    

    时空场景图通过将场景分解为单个对象及其两两时间关系来表示视频中的相互作用。长期预测对象之间精细粒度的两两关系是一个具有挑战性的问题。为此，我们引入了场景图预测（SGA）任务。我们将最先进的场景图生成方法用作基线，以预测对象之间未来的两两关系，并提出了一种新颖的方法SceneSayer。在SceneSayer中，我们利用面向对象的关系表示来推断观察到的视频帧并建模对象之间关系的演变。我们采用连续时间视角，并分别使用神经ODE和神经SDE的概念来建模对象相互作用的潜在动态演变。我们通过解决普通微分方程和随机微分方程来推断未来关系的表示。

    arXiv:2403.04899v1 Announce Type: cross  Abstract: Spatio-temporal scene graphs represent interactions in a video by decomposing scenes into individual objects and their pair-wise temporal relationships. Long-term anticipation of the fine-grained pair-wise relationships between objects is a challenging problem. To this end, we introduce the task of Scene Graph Anticipation (SGA). We adapt state-of-the-art scene graph generation methods as baselines to anticipate future pair-wise relationships between objects and propose a novel approach SceneSayer. In SceneSayer, we leverage object-centric representations of relationships to reason about the observed video frames and model the evolution of relationships between objects. We take a continuous time perspective and model the latent dynamics of the evolution of object interactions using concepts of NeuralODE and NeuralSDE, respectively. We infer representations of future relationships by solving an Ordinary Differential Equation and a Stoch
    
[^6]: 分层复杂度匹配学习产生了改进的大脑皮层V2区模型

    Layerwise complexity-matched learning yields an improved model of cortical area V2

    [https://arxiv.org/abs/2312.11436](https://arxiv.org/abs/2312.11436)

    通过分层复杂度匹配学习，我们开发了一种自下而上的自监督训练方法，最大化了特征相似性同时在不同位置的补丁上解除特征相关性。

    

    人类识别复杂视觉模式的能力是通过顺次区域在腹侧视觉皮层中执行的变换所形成的。最近的端到端训练的深度神经网络逼近了人类的能力，并且提供了迄今为止对层次结构的后期神经反应的最佳描述。然而，与传统的手工设计模型相比，或者与优化编码效率或预测的模型相比，这些网络对前期阶段提供了较差的描述。此外，用于端到端学习的梯度反向传播通常被认为在生物上是不切实际的。在这里，我们通过开发一种自下而上的自监督训练方法，独立地作用于连续层，从而克服了这两个限制。具体地，我们最大化了对局部变形自然图像补丁对之间的特征相似性，并在采样自其他位置的补丁时使特征去相关。

    arXiv:2312.11436v2 Announce Type: replace-cross  Abstract: Human ability to recognize complex visual patterns arises through transformations performed by successive areas in the ventral visual cortex. Deep neural networks trained end-to-end for object recognition approach human capabilities, and offer the best descriptions to date of neural responses in the late stages of the hierarchy. But these networks provide a poor account of the early stages, compared to traditional hand-engineered models, or models optimized for coding efficiency or prediction. Moreover, the gradient backpropagation used in end-to-end learning is generally considered to be biologically implausible. Here, we overcome both of these limitations by developing a bottom-up self-supervised training methodology that operates independently on successive layers. Specifically, we maximize feature similarity between pairs of locally-deformed natural image patches, while decorrelating features across patches sampled from oth
    
[^7]: 快速注册逼真的虚拟现实头像用于面部动画

    Fast Registration of Photorealistic Avatars for VR Facial Animation. (arXiv:2401.11002v1 [cs.CV])

    [http://arxiv.org/abs/2401.11002](http://arxiv.org/abs/2401.11002)

    本论文针对虚拟现实头像注册和面部动画问题，发现头像和头显相机图像之间的领域差距是主要难点，并提出了一个系统设计来解决这个问题。

    

    虚拟现实（VR）在社交互动方面拥有更具沉浸感的潜力。其中一个关键是能够在佩戴VR头显的情况下准确地模拟一个逼真的头像。虽然在离线环境中可以实现对特定个人头像进行高质量注册，并进行动画生成，但通用实时模型的性能明显下降。在线注册也面临诸多挑战，包括倾斜的摄像机视角和不同的模态。在这项工作中，我们首先表明头像与头显相机图像之间的领域差距是困难的主要源泉之一，基于转换器的架构在领域一致的数据上实现了高准确性，在引入领域差距后性能下降。基于此发现，我们提出了一个系统设计，将问题分解为两个部分：1）一个迭代细化模块，接收领域内输入，和2）一个通用的头像引导图像生成模块

    Virtual Reality (VR) bares promise of social interactions that can feel more immersive than other media. Key to this is the ability to accurately animate a photorealistic avatar of one's likeness while wearing a VR headset. Although high quality registration of person-specific avatars to headset-mounted camera (HMC) images is possible in an offline setting, the performance of generic realtime models are significantly degraded. Online registration is also challenging due to oblique camera views and differences in modality. In this work, we first show that the domain gap between the avatar and headset-camera images is one of the primary sources of difficulty, where a transformer-based architecture achieves high accuracy on domain-consistent data, but degrades when the domain-gap is re-introduced. Building on this finding, we develop a system design that decouples the problem into two parts: 1) an iterative refinement module that takes in-domain inputs, and 2) a generic avatar-guided imag
    
[^8]: 物理世界中的对抗样本：一项综述

    Adversarial Examples in the Physical World: A Survey. (arXiv:2311.01473v1 [cs.CV])

    [http://arxiv.org/abs/2311.01473](http://arxiv.org/abs/2311.01473)

    本综述系统地研究了物理世界中的对抗样本（PAEs）的特点，并提出了基于其特征的全面分析和分类框架，涵盖了100多个研究，以填补对PAEs独特特征的现有研究不足。

    

    深度神经网络（DNNs）对对抗样本表现出高度的脆弱性。除了在数字世界中的攻击外，对抗样本在物理世界中的实际影响提出了重大挑战和安全性问题。然而，当前对物理对抗样本（PAEs）的研究缺乏对其独特特征的全面理解，导致其重要性和理解的局限性。本文通过在训练、制造和重采样过程中全面考察PAEs的特点来弥补这一差距。通过分析物理对抗攻击之间的联系，我们确定制造和重采样是PAEs中独特属性和特殊性的主要来源。利用这一知识，我们基于其特定特征开发了一个全面的PAEs分析和分类框架，涵盖了100多个物理对抗世界研究的研究。

    Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial e
    
[^9]: 利用分层特征共享进行高效数据集压缩

    Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation. (arXiv:2310.07506v1 [cs.CV])

    [http://arxiv.org/abs/2310.07506](http://arxiv.org/abs/2310.07506)

    本文提出了一种利用分层特征共享的数据参数化架构（HMN），旨在更高效地压缩数据。通过将数据存储在三层结构中，HMN能够捕捉到数据集级别、类别级别和样本级别的特征。

    

    在真实世界数据集中，数据压缩（DC）旨在合成一个显著较小的数据集，以高性能进行模型训练。最近的研究提出使用数据参数化增强DC，将数据压缩为参数化的数据容器而不是像素空间。数据参数化的直觉是编码图像的共享特征，以避免额外的存储成本。本文认识到由于分类系统的内在分层结构，图像以分层方式共享共同的特征，这是当前数据参数化方法所忽视的。为了更好地使DC与这种分层性质对齐，并在数据容器内部鼓励更高效的信息共享，我们提出了一种新颖的数据参数化架构，分层记忆网络（HMN）。HMN将压缩数据存储在三层结构中，表示数据集级别、类别级别和样本级别的特征。

    Given a real-world dataset, data condensation (DC) aims to synthesize a significantly smaller dataset that captures the knowledge of this dataset for model training with high performance. Recent works propose to enhance DC with data parameterization, which condenses data into parameterized data containers rather than pixel space. The intuition behind data parameterization is to encode shared features of images to avoid additional storage costs. In this paper, we recognize that images share common features in a hierarchical way due to the inherent hierarchical structure of the classification system, which is overlooked by current data parameterization methods. To better align DC with this hierarchical nature and encourage more efficient information sharing inside data containers, we propose a novel data parameterization architecture, Hierarchical Memory Network (HMN). HMN stores condensed data in a three-tier structure, representing the dataset-level, class-level, and instance-level fea
    
[^10]: 数据增强在自动区域性冠状动脉分割中的应用：伪标签法用于疾病诊断

    Data Augmentation through Pseudolabels in Automatic Region Based Coronary Artery Segmentation for Disease Diagnosis. (arXiv:2310.05990v1 [eess.IV])

    [http://arxiv.org/abs/2310.05990](http://arxiv.org/abs/2310.05990)

    这项研究引入了伪标签作为数据增强技术，通过改善基准Yolo模型的性能，提高了冠状动脉分割的效果。

    

    冠状动脉疾病（CAD）是可预防的主要死亡和残疾原因之一。这些疾病的诊断通常困难且资源密集。血管造影图像中的动脉分割已经演变成为一种辅助工具，可以帮助临床医生进行准确的诊断。然而，由于数据量有限且构建数据集的困难，分割任务一直很具挑战性。在本研究中，我们引入了使用伪标签作为数据增强技术来改善基准Yolo模型性能的思想。该方法在验证数据集中将基线的F1分数提高了9％，在测试数据集中提高了3％。

    Coronary Artery Diseases(CADs) though preventable are one of the leading causes of death and disability. Diagnosis of these diseases is often difficult and resource intensive. Segmentation of arteries in angiographic images has evolved as a tool for assistance, helping clinicians in making accurate diagnosis. However, due to the limited amount of data and the difficulty in curating a dataset, the task of segmentation has proven challenging. In this study, we introduce the idea of using pseudolabels as a data augmentation technique to improve the performance of the baseline Yolo model. This method increases the F1 score of the baseline by 9% in the validation dataset and by 3% in the test dataset.
    
[^11]: 面向检测的图像-文本预训练方法用于开放词汇检测

    Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection. (arXiv:2310.00161v1 [cs.CV])

    [http://arxiv.org/abs/2310.00161](http://arxiv.org/abs/2310.00161)

    这项研究提出了一种面向检测的图像-文本预训练方法，旨在弥合图像级预训练和开放词汇目标检测之间的差距。通过检测器架构和对比损失，该方法能够从噪声图像-文本对中学习到新出现的物体-语义线索，并提出了一种平移窗口学习方法来改进主干网络的表示。在LVIS开放词汇检测基准上，该方法取得了显著优于其他方法的40.4的掩码AP$_r$结果。

    

    我们提出了一种基于面向检测的图像-文本预训练的新的开放词汇检测方法，以填补图像级预训练和开放词汇目标检测之间的差距。在预训练阶段，我们用检测器架构替代常用的分类架构，通过使检测器头部能够从噪声图像-文本对中学习，更好地满足检测的区域级识别需求。我们的方法只使用标准的对比损失而不使用伪标签，是对对比学习方法的简单而有效的扩展，可以学习到新出现的物体-语义线索。此外，我们提出了一种基于窗口注意力的平移窗口学习方法，使主干网络的表示更加鲁棒、平移不变，并且不受窗口模式的偏差影响。在流行的LVIS开放词汇检测基准上，我们的方法使用常见的ViT-L主干网络取得了40.4的掩码AP$_r$新的最优结果，明显优于其他方法。

    We present a new open-vocabulary detection approach based on detection-oriented image-text pretraining to bridge the gap between image-level pretraining and open-vocabulary object detection. At the pretraining phase, we replace the commonly used classification architecture with the detector architecture, which better serves the region-level recognition needs of detection by enabling the detector heads to learn from noisy image-text pairs. Using only standard contrastive loss and no pseudo-labeling, our approach is a simple yet effective extension of the contrastive learning method to learn emergent object-semantic cues. In addition, we propose a shifted-window learning approach upon window attention to make the backbone representation more robust, translation-invariant, and less biased by the window pattern. On the popular LVIS open-vocabulary detection benchmark, our approach sets a new state of the art of 40.4 mask AP$_r$ using the common ViT-L backbone, significantly outperforming t
    
[^12]: 交叉验证就是你所需的：一种统计方法来估计标签噪声。

    Cross-Validation Is All You Need: A Statistical Approach To Label Noise Estimation. (arXiv:2306.13990v1 [cs.LG])

    [http://arxiv.org/abs/2306.13990](http://arxiv.org/abs/2306.13990)

    本论文提出了一种重复交叉验证(Repeated Cross-Validation)方法，通过构建噪声直方图并提出三种基于该直方图的方法来检测标签噪声并清理数据，解决了结果预测分析中的数据清洗问题。

    

    标签噪声在机器学习数据集中普遍存在。鉴定和消除标签噪声至关重要，因为在噪声数据上训练的模型会大幅降低准确性和泛化性。大多数现有的标签噪声检测方法都是为分类任务设计的，而基于结果预测分析的数据清理相对未被探索。受到交叉验证中不同折的性能波动的启发，我们提出了用于标签噪声估计的重复交叉验证（ReCoV）来填补这一空白。ReCoV通过记录每个最差表现折中的样本ID来构建一个噪声直方图，以此来排名样本的噪声水平。我们进一步提出了三种基于噪声直方图来鉴别嘈杂样本的方法，以解决越来越复杂的噪声分布。我们展示了ReCoV在分类任务基准中的优越表现，优于现有最先进标签清理算法。更重要的是，

    Label noise is prevalent in machine learning datasets. It is crucial to identify and remove label noise because models trained on noisy data can have substantially reduced accuracy and generalizability. Most existing label noise detection approaches are designed for classification tasks, and data cleaning for outcome prediction analysis is relatively unexplored. Inspired by the fluctuations in performance across different folds in cross-validation, we propose Repeated Cross-Validations for label noise estimation (ReCoV) to address this gap. ReCoV constructs a noise histogram that ranks the noise level of samples based on a large number of cross-validations by recording sample IDs in each worst-performing fold. We further propose three approaches for identifying noisy samples based on noise histograms to address increasingly complex noise distributions. We show that ReCoV outperforms state-of-the-art algorithms for label cleaning in a classification task benchmark. More importantly, we 
    

