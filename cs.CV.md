# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data](https://arxiv.org/abs/2403.16582) | 研究调查了在全球范围内同时选择融合策略和编码器架构对作物分类具有的影响。 |
| [^2] | [SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces](https://arxiv.org/abs/2403.07711) | 提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题 |
| [^3] | [Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data.](http://arxiv.org/abs/2401.15113) | 本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。 |
| [^4] | [Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper.](http://arxiv.org/abs/2308.13495) | 本论文提出了一个仿制谷歌眼动论文的开源实现，重点是通过整合机器学习技术，在智能手机上实现与谷歌论文相当的准确眼动追踪解决方案。 |
| [^5] | [Model-agnostic explainable artificial intelligence for object detection in image data.](http://arxiv.org/abs/2303.17249) | 本文设计并实现了一种新的黑盒解释方法——BODEM，它采用了局部和远程掩蔽生成多个版本的输入图像，从而比目前用于解释对象检测的其他三种最先进的方法提供更详细和有用的解释。 |
| [^6] | [FaceRNET: a Facial Expression Intensity Estimation Network.](http://arxiv.org/abs/2303.00180) | 本文介绍了一种名为FaceRNET的面部表情强度估计网络，该网络采用了表示提取器和循环神经网络的组合，能够从视频中提取各种情感描述符，并通过动态路由处理不同长度的输入视频。在Hume-Reaction数据集上进行的测试表明，该方法取得了优秀的结果。 |

# 详细

[^1]: 在利用全球遥感数据进行作物分类的多视图学习模型的最佳选择研究

    In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data

    [https://arxiv.org/abs/2403.16582](https://arxiv.org/abs/2403.16582)

    研究调查了在全球范围内同时选择融合策略和编码器架构对作物分类具有的影响。

    

    作物分类在研究作物模式变化、资源管理和碳固存中具有至关重要的作用。采用数据驱动技术进行预测时，利用各种时间数据源是必要的。深度学习模型已被证明对将时间序列数据映射到高级表示以进行预测任务非常有效。然而，当处理多个输入模式时，它们面临着重大挑战。文献对多视图学习（MVL）场景提供了有限的指导，主要集中在探索具有特定编码器的融合策略，并在局部地区对其进行验证。相反，我们研究了在全球范围内对农田土地和作物类型进行分类时同时选择融合策略和编码器架构的影响。

    arXiv:2403.16582v1 Announce Type: cross  Abstract: Crop classification is of critical importance due to its role in studying crop pattern changes, resource management, and carbon sequestration. When employing data-driven techniques for its prediction, utilizing various temporal data sources is necessary. Deep learning models have proven to be effective for this task by mapping time series data to high-level representation for prediction. However, they face substantial challenges when dealing with multiple input patterns. The literature offers limited guidance for Multi-View Learning (MVL) scenarios, as it has primarily focused on exploring fusion strategies with specific encoders and validating them in local regions. In contrast, we investigate the impact of simultaneous selection of the fusion strategy and the encoder architecture evaluated on a global-scale cropland and crop-type classifications. We use a range of five fusion strategies (Input, Feature, Decision, Ensemble, Hybrid) an
    
[^2]: SSM遇上视频扩散模型: 结构化状态空间下的高效视频生成

    SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces

    [https://arxiv.org/abs/2403.07711](https://arxiv.org/abs/2403.07711)

    提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题

    

    鉴于图像生成通过扩散模型取得的显著成就，研究界对将这些模型扩展到视频生成表现出越来越大的兴趣。最近用于视频生成的扩散模型主要利用注意力层来提取时间特征。然而，由于注意力层的内存消耗随着序列长度的增加呈二次增长，这种限制在尝试使用扩散模型生成更长视频序列时会带来重大挑战。为了克服这一挑战，我们提出利用状态空间模型（SSMs）。由于相对于序列长度，SSMs具有线性内存消耗，最近已经引起了越来越多的关注。在实验中，我们首先通过使用UCF101这一视频生成的标准基准来评估我们基于SSM的模型。此外，为探讨SSMs在更长视频生成中的潜力，

    arXiv:2403.07711v1 Announce Type: cross  Abstract: Given the remarkable achievements in image generation through diffusion models, the research community has shown increasing interest in extending these models to video generation. Recent diffusion models for video generation have predominantly utilized attention layers to extract temporal features. However, attention layers are limited by their memory consumption, which increases quadratically with the length of the sequence. This limitation presents significant challenges when attempting to generate longer video sequences using diffusion models. To overcome this challenge, we propose leveraging state-space models (SSMs). SSMs have recently gained attention as viable alternatives due to their linear memory consumption relative to sequence length. In the experiments, we first evaluate our SSM-based model with UCF101, a standard benchmark of video generation. In addition, to investigate the potential of SSMs for longer video generation, 
    
[^3]: 使用深度学习和开放地球观测数据实现全球冰川制图

    Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data. (arXiv:2401.15113v1 [cs.CV])

    [http://arxiv.org/abs/2401.15113](http://arxiv.org/abs/2401.15113)

    本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。

    

    准确的全球冰川制图对于理解气候变化的影响至关重要。这个过程受到冰川多样性、难以分类的碎石和大数据处理的挑战。本文提出了Glacier-VisionTransformer-U-Net (GlaViTU)，一个卷积-Transformer深度学习模型，并提出了五种利用开放卫星影像进行多时相全球冰川制图的策略。空间、时间和跨传感器的泛化性能评估表明，我们的最佳策略在大多数情况下实现了IoU（交并比）> 0.85，并且在以冰雪为主的地区增加到了> 0.90，而在高山亚洲等碎石丰富的区域则降至> 0.75。此外，添加合成孔径雷达数据，即回波和干涉相干度，可以提高所有可用地区的准确性。报告冰川范围的校准置信度使预测更可靠和可解释。我们还发布了一个基准数据集。

    Accurate global glacier mapping is critical for understanding climate change impacts. It is challenged by glacier diversity, difficult-to-classify debris and big data processing. Here we propose Glacier-VisionTransformer-U-Net (GlaViTU), a convolutional-transformer deep learning model, and five strategies for multitemporal global-scale glacier mapping using open satellite imagery. Assessing the spatial, temporal and cross-sensor generalisation shows that our best strategy achieves intersection over union >0.85 on previously unobserved images in most cases, which drops to >0.75 for debris-rich areas such as High-Mountain Asia and increases to >0.90 for regions dominated by clean ice. Additionally, adding synthetic aperture radar data, namely, backscatter and interferometric coherence, increases the accuracy in all regions where available. The calibrated confidence for glacier extents is reported making the predictions more reliable and interpretable. We also release a benchmark dataset 
    
[^4]: 开放注视：一个仿制谷歌眼动论文的开源实现

    Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper. (arXiv:2308.13495v1 [cs.CV])

    [http://arxiv.org/abs/2308.13495](http://arxiv.org/abs/2308.13495)

    本论文提出了一个仿制谷歌眼动论文的开源实现，重点是通过整合机器学习技术，在智能手机上实现与谷歌论文相当的准确眼动追踪解决方案。

    

    眼动已经成为视觉研究、语言分析和可用性评估等不同领域的重要工具。然而，大多数先前的研究集中在使用专门的、昂贵的眼动追踪硬件的扩展式桌面显示器上。尽管智能手机的普及率和使用频率很高，但对于智能手机上的眼球移动模式却鲜有见解。在本文中，我们提出了一个基于智能手机的开源注视追踪实现，模拟了谷歌论文提出的方法论（其源代码仍然是专有的）。我们的重点是在不需要额外硬件的情况下达到与谷歌论文方法相当的准确度。通过整合机器学习技术，我们揭示了一种本地于智能手机的准确眼动追踪解决方案。我们的方法展示了与最先进的移动眼动追踪器相当的精度。

    Eye tracking has been a pivotal tool in diverse fields such as vision research, language analysis, and usability assessment. The majority of prior investigations, however, have concentrated on expansive desktop displays employing specialized, costly eye tracking hardware that lacks scalability. Remarkably little insight exists into ocular movement patterns on smartphones, despite their widespread adoption and significant usage. In this manuscript, we present an open-source implementation of a smartphone-based gaze tracker that emulates the methodology proposed by a GooglePaper (whose source code remains proprietary). Our focus is on attaining accuracy comparable to that attained through the GooglePaper's methodology, without the necessity for supplementary hardware. Through the integration of machine learning techniques, we unveil an accurate eye tracking solution that is native to smartphones. Our approach demonstrates precision akin to the state-of-the-art mobile eye trackers, which 
    
[^5]: 面向对象检测的模型无关可解释人工智能

    Model-agnostic explainable artificial intelligence for object detection in image data. (arXiv:2303.17249v1 [cs.CV])

    [http://arxiv.org/abs/2303.17249](http://arxiv.org/abs/2303.17249)

    本文设计并实现了一种新的黑盒解释方法——BODEM，它采用了局部和远程掩蔽生成多个版本的输入图像，从而比目前用于解释对象检测的其他三种最先进的方法提供更详细和有用的解释。

    

    对象检测是计算机视觉中的基本任务之一，通过开发大型复杂的深度学习模型已经取得了很大进展。然而，缺乏透明度是一个重要的挑战，可能妨碍这些模型的广泛应用。可解释的人工智能是一个研究领域，其中开发方法来帮助用户理解基于人工智能的系统的行为、决策逻辑和漏洞。本文为了解释基于人工智能的对象检测系统设计和实现了一种名为Black-box Object Detection Explanation by Masking（BODEM）的黑盒说明方法，采用新的掩蔽方法。我们提出了局部和远程掩蔽来生成输入图像的多个版本。局部掩蔽用于干扰目标对象内的像素，以了解对象检测器对这些变化的反应，而远程掩蔽则用于研究对象检测器在图像背景上的行为。我们在三个基准数据集上的实验表明，与用于解释对象检测的其他三种最先进的方法相比，BODEM提供了更详细和有用的说明。

    Object detection is a fundamental task in computer vision, which has been greatly progressed through developing large and intricate deep learning models. However, the lack of transparency is a big challenge that may not allow the widespread adoption of these models. Explainable artificial intelligence is a field of research where methods are developed to help users understand the behavior, decision logics, and vulnerabilities of AI-based systems. Black-box explanation refers to explaining decisions of an AI system without having access to its internals. In this paper, we design and implement a black-box explanation method named Black-box Object Detection Explanation by Masking (BODEM) through adopting a new masking approach for AI-based object detection systems. We propose local and distant masking to generate multiple versions of an input image. Local masks are used to disturb pixels within a target object to figure out how the object detector reacts to these changes, while distant ma
    
[^6]: FaceRNET: 一种面部表情强度估计网络

    FaceRNET: a Facial Expression Intensity Estimation Network. (arXiv:2303.00180v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.00180](http://arxiv.org/abs/2303.00180)

    本文介绍了一种名为FaceRNET的面部表情强度估计网络，该网络采用了表示提取器和循环神经网络的组合，能够从视频中提取各种情感描述符，并通过动态路由处理不同长度的输入视频。在Hume-Reaction数据集上进行的测试表明，该方法取得了优秀的结果。

    

    本文介绍了我们在视频中进行面部表情强度估计的方法。它包括两个组件：i) 一个表示提取器网络，从每个视频帧中提取各种情感描述符（价值-唤醒、动作单元和基本表情）；ii) 一个循环神经网络（RNN），捕捉数据中的时间信息，然后是一个掩码层，通过动态路由实现对不同输入视频长度的处理能力。该方法在Hume-Reaction数据集上进行了测试，并取得了出色的结果。

    This paper presents our approach for Facial Expression Intensity Estimation from videos. It includes two components: i) a representation extractor network that extracts various emotion descriptors (valence-arousal, action units and basic expressions) from each videoframe; ii) a RNN that captures temporal information in the data, followed by a mask layer which enables handling varying input video lengths through dynamic routing. This approach has been tested on the Hume-Reaction dataset yielding excellent results.
    

