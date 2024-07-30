# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery](https://arxiv.org/abs/2404.02877) | 本研究对卫星图像中识别飞机的任务自定义的一套先进对象检测算法进行了全面评估和比较，发现YOLOv5是在不同成像条件下展现高精度和适应性的最优模型。 |
| [^2] | [SportsNGEN: Sustained Generation of Multi-player Sports Gameplay](https://arxiv.org/abs/2403.12977) | SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。 |
| [^3] | [N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields](https://arxiv.org/abs/2403.10997) | 利用Nested Neural Feature Fields (N2F2) 实现了层次化监督学习，提供了对物理维度或语义维度等不同粒度的场景属性全面和细致的理解。 |
| [^4] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^5] | [Weighted Ensemble Models Are Strong Continual Learners](https://arxiv.org/abs/2312.08977) | 通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。 |
| [^6] | [Adaptive Self-training Framework for Fine-grained Scene Graph Generation.](http://arxiv.org/abs/2401.09786) | 本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。 |
| [^7] | [CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians.](http://arxiv.org/abs/2401.05925) | CoSSegGaussians是一种紧凑且迅速的3D高斯场景分割方法，通过映射空间和语义特征实现紧凑和可靠的零样本场景分割。 |
| [^8] | [Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation.](http://arxiv.org/abs/2310.03986) | 通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。 |
| [^9] | [MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool.](http://arxiv.org/abs/2309.16701) | 本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。 |
| [^10] | [GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields.](http://arxiv.org/abs/2308.16891) | GNFactor是一个用于多任务机器人操作的代理方法，它利用可泛化神经特征场和Perceiver Transformer模块，以及深度三维体素表示来实现对真实世界环境中的操作任务的执行。它通过将视觉和语义信息纳入三维表示来提高场景的理解能力，并在多个任务上进行了验证。 |
| [^11] | [Explainable Multi-View Deep Networks Methodology for Experimental Physics.](http://arxiv.org/abs/2308.08206) | 该论文介绍了一个可解释的多视角深度网络方法论，应用于实验物理中的多种成像表达分析。该方法论解决了多视角模型可解释性不足的问题。 |
| [^12] | [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?.](http://arxiv.org/abs/2306.06048) | 本研究旨在探究微调对少样本下游任务的外分布检测的影响，发现适当选择外分布分数对于CLIP-based 微调至关重要。最大概念匹配（MCM）分数提供了一个有前途的解决方案。 |
| [^13] | [ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation.](http://arxiv.org/abs/2305.01618) | 本研究提出了一种基于视觉遥操作的数据收集方法以及学习手物互动先验的新方法，从而能够在联结目标和手部姿态估计中实现更好的关键点定位性能。 |
| [^14] | [Differentiable Gaussianization Layers for Inverse Problems Regularized by Deep Generative Models.](http://arxiv.org/abs/2112.03860) | 该论文提出了一种使用可微数据相关层进行重新参数化和高斯化潜在张量的方法，以约束逆问题为获得高保真度的分布内解决方案，有效解决深度生成模型逆问题中潜在张量偏离期望高斯分布的问题。 |

# 详细

[^1]: FlightScope: 卫星图像中飞行器检测算法的深度全面评估

    FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery

    [https://arxiv.org/abs/2404.02877](https://arxiv.org/abs/2404.02877)

    本研究对卫星图像中识别飞机的任务自定义的一套先进对象检测算法进行了全面评估和比较，发现YOLOv5是在不同成像条件下展现高精度和适应性的最优模型。

    

    arXiv:2404.02877v1 公告类型：跨领域 摘要：在遥感卫星图像中进行对象检测对于许多领域，如生物物理学和环境监测至关重要。尽管深度学习算法不断发展，但它们大多在常见的基于地面拍摄的照片上实施和测试。本文对一套针对在卫星图像中识别飞机这一任务定制的先进对象检测算法进行了批判性评估和比较。利用大型HRPlanesV2数据集，以及与GDIT数据集的严格验证，该研究涵盖了一系列方法，包括YOLO版本5和8、Faster RCNN、CenterNet、RetinaNet、RTMDet和DETR，均是从头开始训练的。这项全面的训练和验证研究揭示了YOLOv5作为识别遥感数据中的飞机这一特定案例的卓越模型，展示了其在不同成像条件下的高精度和适应性。

    arXiv:2404.02877v1 Announce Type: cross  Abstract: Object detection in remotely sensed satellite pictures is fundamental in many fields such as biophysical, and environmental monitoring. While deep learning algorithms are constantly evolving, they have been mostly implemented and tested on popular ground-based taken photos. This paper critically evaluates and compares a suite of advanced object detection algorithms customized for the task of identifying aircraft within satellite imagery. Using the large HRPlanesV2 dataset, together with a rigorous validation with the GDIT dataset, this research encompasses an array of methodologies including YOLO versions 5 and 8, Faster RCNN, CenterNet, RetinaNet, RTMDet, and DETR, all trained from scratch. This exhaustive training and validation study reveal YOLOv5 as the preeminent model for the specific case of identifying airplanes from remote sensing data, showcasing high precision and adaptability across diverse imaging conditions. This research
    
[^2]: SportsNGEN: 持续生成多人体育游戏

    SportsNGEN: Sustained Generation of Multi-player Sports Gameplay

    [https://arxiv.org/abs/2403.12977](https://arxiv.org/abs/2403.12977)

    SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。

    

    我们提出了一种基于Transformer解码器的模型SportsNGEN，该模型经过训练使用运动员和球追踪序列，能够生成逼真且持续的游戏场景。我们在大量专业网球追踪数据上训练和评估SportsNGEN，并展示通过将生成的模拟与射击分类器和逻辑相结合来开始和结束球赛，系统能够模拟整个网球比赛。此外，SportsNGEN的通用版本可以通过在包含该球员的比赛数据上微调来定制特定球员。我们展示了我们的模型经过良好校准，可以通过评估反事实或假设选项为教练和广播员提供洞察力。最后，我们展示了质量结果表明相同的方法适用于足球。

    arXiv:2403.12977v1 Announce Type: cross  Abstract: We present a transformer decoder based model, SportsNGEN, that is trained on sports player and ball tracking sequences that is capable of generating realistic and sustained gameplay. We train and evaluate SportsNGEN on a large database of professional tennis tracking data and demonstrate that by combining the generated simulations with a shot classifier and logic to start and end rallies, the system is capable of simulating an entire tennis match. In addition, a generic version of SportsNGEN can be customized to a specific player by fine-tuning on match data that includes that player. We show that our model is well calibrated and can be used to derive insights for coaches and broadcasters by evaluating counterfactual or what if options. Finally, we show qualitative results indicating the same approach works for football.
    
[^3]: 嵌套神经特征场的层次场景理解

    N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields

    [https://arxiv.org/abs/2403.10997](https://arxiv.org/abs/2403.10997)

    利用Nested Neural Feature Fields (N2F2) 实现了层次化监督学习，提供了对物理维度或语义维度等不同粒度的场景属性全面和细致的理解。

    

    在计算机视觉中，理解多层抽象的复杂场景仍然是一个巨大挑战。为了解决这个问题，我们引入了嵌套神经特征场 (N2F2)，这是一种新颖的方法，利用分层监督来学习单个特征场，在同一高维特征中的不同维度编码不同粒度的场景属性。我们的方法允许灵活定义层次，可以根据物理维度、语义维度或两者均匹配，从而实现对场景的全面和细致理解。我们利用2D类别无关分割模型在图像空间的任意尺度提供语义有意义的像素分组，并查询CLIP视觉编码器，为这些段落中的每个部分获得与语言对齐的嵌入。我们提出的分层监督方法将不同的嵌套特征场维度分配给提取C

    arXiv:2403.10997v1 Announce Type: cross  Abstract: Understanding complex scenes at multiple levels of abstraction remains a formidable challenge in computer vision. To address this, we introduce Nested Neural Feature Fields (N2F2), a novel approach that employs hierarchical supervision to learn a single feature field, wherein different dimensions within the same high-dimensional feature encode scene properties at varying granularities. Our method allows for a flexible definition of hierarchies, tailored to either the physical dimensions or semantics or both, thereby enabling a comprehensive and nuanced understanding of scenes. We leverage a 2D class-agnostic segmentation model to provide semantically meaningful pixel groupings at arbitrary scales in the image space, and query the CLIP vision-encoder to obtain language-aligned embeddings for each of these segments. Our proposed hierarchical supervision method then assigns different nested dimensions of the feature field to distill the C
    
[^4]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^5]: 加权集成模型是强大的持续学习者

    Weighted Ensemble Models Are Strong Continual Learners

    [https://arxiv.org/abs/2312.08977](https://arxiv.org/abs/2312.08977)

    通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。

    

    在本文中，我们研究持续学习（CL）的问题，其中目标是从一系列任务中学习模型，使得以前任务的数据在学习当前任务数据时不可用。CL本质上是在能够学习新任务（即可塑性）和保持先前学习概念的性能（即稳定性）之间取得平衡的过程。为了解决稳定性-可塑性的权衡问题，我们建议对先前和当前任务的模型参数进行加权集成。这种加权集成模型，我们称之为持续模型平均（或CoMA），通过利用可塑性在当前任务上获得高准确性，同时不会偏离太远的先前权重配置，从而确保稳定性。我们还提出了CoMA的改进型变体，名为持续费舍尔加权模型平均（或CoFiMA），该模型对每一个参数进行选择性加权。

    arXiv:2312.08977v2 Announce Type: replace-cross  Abstract: In this work, we study the problem of continual learning (CL) where the goal is to learn a model on a sequence of tasks, such that the data from the previous tasks becomes unavailable while learning on the current task data. CL is essentially a balancing act between being able to learn on the new task (i.e., plasticity) and maintaining the performance on the previously learned concepts (i.e., stability). Intending to address the stability-plasticity trade-off, we propose to perform weight-ensembling of the model parameters of the previous and current tasks. This weighted-ensembled model, which we call Continual Model Averaging (or CoMA), attains high accuracy on the current task by leveraging plasticity, while not deviating too far from the previous weight configuration, ensuring stability. We also propose an improved variant of CoMA, named Continual Fisher-weighted Model Averaging (or CoFiMA), that selectively weighs each para
    
[^6]: 自适应自训练框架用于细粒度场景图生成

    Adaptive Self-training Framework for Fine-grained Scene Graph Generation. (arXiv:2401.09786v1 [cs.CV])

    [http://arxiv.org/abs/2401.09786](http://arxiv.org/abs/2401.09786)

    本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。

    

    场景图生成（SGG）模型在基准数据集中存在长尾谓词分布和缺失注释问题。本研究旨在通过利用未标注的三元组缓解SGG的长尾问题。为此，我们引入了一种称为自训练SGG（ST-SGG）的框架，该框架基于未标注的三元组为其分配伪标签以训练SGG模型。虽然在图像识别方面的自训练取得了显著进展，但设计适用于SGG任务的自训练框架更具挑战，因为其固有特性，如语义歧义和长尾分布的谓词类别。因此，我们提出了一种新颖的SGG伪标签技术，称为具有动量的类别自适应阈值化（CATM），它是一种独立于模型的框架，可应用于任何已有的SGG模型。此外，我们设计了一个图结构学习器（GSL），从中获益。

    Scene graph generation (SGG) models have suffered from inherent problems regarding the benchmark datasets such as the long-tailed predicate distribution and missing annotation problems. In this work, we aim to alleviate the long-tailed problem of SGG by utilizing unannotated triplets. To this end, we introduce a Self-Training framework for SGG (ST-SGG) that assigns pseudo-labels for unannotated triplets based on which the SGG models are trained. While there has been significant progress in self-training for image recognition, designing a self-training framework for the SGG task is more challenging due to its inherent nature such as the semantic ambiguity and the long-tailed distribution of predicate classes. Hence, we propose a novel pseudo-labeling technique for SGG, called Class-specific Adaptive Thresholding with Momentum (CATM), which is a model-agnostic framework that can be applied to any existing SGG models. Furthermore, we devise a graph structure learner (GSL) that is benefici
    
[^7]: CoSSegGaussians：紧凑且迅速的3D高斯场景分割方法

    CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians. (arXiv:2401.05925v1 [cs.CV])

    [http://arxiv.org/abs/2401.05925](http://arxiv.org/abs/2401.05925)

    CoSSegGaussians是一种紧凑且迅速的3D高斯场景分割方法，通过映射空间和语义特征实现紧凑和可靠的零样本场景分割。

    

    我们提出了一种紧凑且迅速的3D高斯场景分割方法（CoSSegGaussians），该方法仅使用RGB图像输入，以快速的渲染速度实现紧凑的3D一致性场景分割。先前基于NeRF的3D分割方法依赖于隐式或体素神经场表示和光线行进体积渲染，这些方法耗时较长。最近的3D高斯场投影显著提高了渲染速度，然而，现有的基于高斯的分割方法（例如高斯分组）在零样本分割中没有提供紧凑的分割掩模，主要原因是在遇到不一致的2D机器生成标签时，无法直接为每个高斯分配可学习参数，缺乏鲁棒性和紧凑性。我们的方法旨在通过使用浅层解码网络将每个高斯点的融合空间和语义上有意义的特征映射，迅速实现紧凑且可靠的零样本场景分割。

    We propose Compact and Swift Segmenting 3D Gaussians(CoSSegGaussians), a method for compact 3D-consistent scene segmentation at fast rendering speed with only RGB images input. Previous NeRF-based 3D segmentation methods have relied on implicit or voxel neural scene representation and ray-marching volume rendering which are time consuming. Recent 3D Gaussian Splatting significantly improves the rendering speed, however, existing Gaussians-based segmentation methods(eg: Gaussian Grouping) fail to provide compact segmentation masks especially in zero-shot segmentation, which is mainly caused by the lack of robustness and compactness for straightforwardly assigning learnable parameters to each Gaussian when encountering inconsistent 2D machine-generated labels. Our method aims to achieve compact and reliable zero-shot scene segmentation swiftly by mapping fused spatial and semantically meaningful features for each Gaussian point with a shallow decoding network. Specifically, our method fi
    
[^8]: 通过参数高效适应，实现对缺失模态的鲁棒多模态学习

    Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation. (arXiv:2310.03986v1 [cs.CV])

    [http://arxiv.org/abs/2310.03986](http://arxiv.org/abs/2310.03986)

    通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。

    

    多模态学习旨在利用多个数据源来提高下游任务的整体性能。在一些相关的模态中观察到，如果在测试时间缺少一个或多个模态，现有的多模态网络的性能会显著下降。为了实现对缺失模态的鲁棒性，我们提出了预训练的多模态网络的简单和参数高效的适应程序。特别地，我们利用低秩适应和中间特征的调制来补偿缺失的模态。我们证明，这种适应可以部分弥补由于缺失模态而导致的性能下降，并在某些情况下胜过针对可用模态组合进行训练的独立的、专门的网络。所提出的适应所需的参数非常少（例如，少于）

    Multimodal learning seeks to utilize data from multiple sources to improve the overall performance of downstream tasks. It is desirable for redundancies in the data to make multimodal systems robust to missing or corrupted observations in some correlated modalities. However, we observe that the performance of several existing multimodal networks significantly deteriorates if one or multiple modalities are absent at test time. To enable robustness to missing modalities, we propose simple and parameter-efficient adaptation procedures for pretrained multimodal networks. In particular, we exploit low-rank adaptation and modulation of intermediate features to compensate for the missing modalities. We demonstrate that such adaptation can partially bridge performance drop due to missing modalities and outperform independent, dedicated networks trained for the available modality combinations in some cases. The proposed adaptation requires extremely small number of parameters (e.g., fewer than 
    
[^9]: MVMR: 在多个可靠视频集中评估自然语言视频定位偏差

    MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool. (arXiv:2309.16701v1 [cs.CV])

    [http://arxiv.org/abs/2309.16701](http://arxiv.org/abs/2309.16701)

    本文提出了一个名为MVMR的任务，旨在给定文本查询从大量视频集中定位视频帧。我们通过已有数据集进行相似性筛选来构建数据集，并引入三个MVMR数据集。我们采用了嵌入式文本相似度匹配和视频-语言对齐技术来计算相关性得分，并为MVMR任务开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    

    随着近年来多媒体内容的激增，自然语言视频定位成为一个关键问题，它致力于检测与给定自然语言查询匹配的视频片段。然而，以往的研究都没有探索在存在多个正负视频的大量语料库中定位一个时刻。本文提出了一个名为MVMR（Massive Videos Moment Retrieval）的任务，旨在给定文本查询从大量视频集中定位视频帧。对于这个任务，我们提出了一种通过对现有视频定位数据集进行相似性筛选来构建数据集的方法，并引入了三个MVMR数据集。具体来说，我们采用基于嵌入的文本相似度匹配和视频-语言对齐技术来计算目标查询与视频之间的相关性得分，从而定义正负集。针对提出的MVMR任务，我们进一步开发了一个强大的模型，Reliable Mutual Matching Network (RMMN)。

    With the explosion of multimedia content in recent years, natural language video localization, which focuses on detecting video moment that matches a given natural language query, has become a critical problem. However, none of the previous research explores localizing a moment from a large corpus where multiple positive and negative videos exist. In this paper, we propose an MVMR (Massive Videos Moment Retrieval) task, which aims to localize video frames from a massive set of videos given a text query. For this task, we suggest methods for constructing datasets by employing similarity filtering on the existing video localization datasets and introduce three MVMR datasets. Specifically, we employ embedding-based text similarity matching and video-language grounding techniques to calculate the relevance score between a target query and videos to define positive and negative sets. For the proposed MVMR task, we further develop a strong model, Reliable Mutual Matching Network (RMMN), whic
    
[^10]: GNFactor：具有可泛化神经特征场的多任务真实机器人学习

    GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields. (arXiv:2308.16891v1 [cs.RO])

    [http://arxiv.org/abs/2308.16891](http://arxiv.org/abs/2308.16891)

    GNFactor是一个用于多任务机器人操作的代理方法，它利用可泛化神经特征场和Perceiver Transformer模块，以及深度三维体素表示来实现对真实世界环境中的操作任务的执行。它通过将视觉和语义信息纳入三维表示来提高场景的理解能力，并在多个任务上进行了验证。

    

    在无结构的现实世界环境中，从视觉观察中开发能够执行多样化操作任务的代理机器人一直是机器人学中的一个长期问题。为了实现这个目标，机器人需要全面理解场景的三维结构和语义。在这项工作中，我们提出了GNFactor，一种用于多任务机器人操作的可视行为克隆代理，它利用可泛化神经特征场（GNF）作为重建模块，Perceiver Transformer作为决策模块，共享深度三维体素表示。为了将语义纳入三维表示，重建模块利用视觉语言基础模型（例如，稳定扩散）将丰富的语义信息提取到深度三维体素中。我们在3个真实机器人任务上评估了GNFactor，并对10个RLBench任务进行了详细的消融实验，只使用了有限数量的数据。

    It is a long-standing problem in robotics to develop agents capable of executing diverse manipulation tasks from visual observations in unstructured real-world environments. To achieve this goal, the robot needs to have a comprehensive understanding of the 3D structure and semantics of the scene. In this work, we present $\textbf{GNFactor}$, a visual behavior cloning agent for multi-task robotic manipulation with $\textbf{G}$eneralizable $\textbf{N}$eural feature $\textbf{F}$ields. GNFactor jointly optimizes a generalizable neural field (GNF) as a reconstruction module and a Perceiver Transformer as a decision-making module, leveraging a shared deep 3D voxel representation. To incorporate semantics in 3D, the reconstruction module utilizes a vision-language foundation model ($\textit{e.g.}$, Stable Diffusion) to distill rich semantic information into the deep 3D voxel. We evaluate GNFactor on 3 real robot tasks and perform detailed ablations on 10 RLBench tasks with a limited number of
    
[^11]: 可解释的多视角深度网络方法论在实验物理中的应用

    Explainable Multi-View Deep Networks Methodology for Experimental Physics. (arXiv:2308.08206v1 [cs.CV])

    [http://arxiv.org/abs/2308.08206](http://arxiv.org/abs/2308.08206)

    该论文介绍了一个可解释的多视角深度网络方法论，应用于实验物理中的多种成像表达分析。该方法论解决了多视角模型可解释性不足的问题。

    

    物理实验常涉及多种成像表达，如X射线扫描和显微图像。深度学习模型已广泛应用于这些实验的监督分析中。合并不同的图像表达经常需要正确分析和做出决策。因此，多视角数据应运而生 - 数据集中的每个样本由来自不同角度、来源或模态的视图描述。多视角学习的概念解决了这些问题。理解深度学习模型的决策过程对于可靠和可信的分析至关重要。因此，最近提出了许多可解释性方法。然而，多视角模型缺乏适当的可解释性，由于其架构的复杂性，难以解释。在本文中，我们提出了适用于视觉领域的不同多视角架构，每个架构都适合解决不同的问题，并提出了解释多视角模型的方法论。

    Physical experiments often involve multiple imaging representations, such as X-ray scans and microscopic images. Deep learning models have been widely used for supervised analysis in these experiments. Combining different image representations is frequently required to analyze and make a decision properly. Consequently, multi-view data has emerged - datasets where each sample is described by views from different angles, sources, or modalities. These problems are addressed with the concept of multi-view learning. Understanding the decision-making process of deep learning models is essential for reliable and credible analysis. Hence, many explainability methods have been devised recently. Nonetheless, there is a lack of proper explainability in multi-view models, which are challenging to explain due to their architectures. In this paper, we suggest different multi-view architectures for the vision domain, each suited to another problem, and we also present a methodology for explaining th
    
[^12]: 微调对于视觉语言模型外分布检测的影响是怎样的？

    How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?. (arXiv:2306.06048v1 [cs.CV])

    [http://arxiv.org/abs/2306.06048](http://arxiv.org/abs/2306.06048)

    本研究旨在探究微调对少样本下游任务的外分布检测的影响，发现适当选择外分布分数对于CLIP-based 微调至关重要。最大概念匹配（MCM）分数提供了一个有前途的解决方案。

    

    最近的大型视觉语言模型，如CLIP，在外分布检测和泛化性能方面表现出色。然而，它们的零样本内分布准确性往往在下游数据集中受到限制。最近的基于CLIP的微调方法，如提示学习，已经在存在外分布标签的情况下显著改进了内分布分类和外分布泛化。然而，模型对于没有外分布标签的语义转移是否可靠仍然不清楚。为了填补这一空白，本文旨在对微调对于少样本下游任务的外分布检测的影响进行全面研究。通过将外分布检测框架化为多模式概念匹配，我们建立了微调方法和各种外分布分数之间的联系。我们的结果表明，选择适当的外分布分数对于基于CLIP的微调至关重要。特别是，最大概念匹配（MCM）分数提供了一个有前途的解决方案。

    Recent large vision-language models such as CLIP have shown remarkable out-of-distribution (OOD) detection and generalization performance. However, their zero-shot in-distribution (ID) accuracy is often limited for downstream datasets. Recent CLIP-based fine-tuning methods such as prompt learning have demonstrated significant improvements in ID classification and OOD generalization where OOD labels are available. Nonetheless, it remains unclear whether the model is reliable to semantic shifts without OOD labels. In this paper, we aim to bridge the gap and present a comprehensive study to understand how fine-tuning impact OOD detection for few-shot downstream tasks. By framing OOD detection as multi-modal concept matching, we establish a connection between fine-tuning methods and various OOD scores. Our results suggest that a proper choice of OOD scores is essential for CLIP-based fine-tuning. In particular, the maximum concept matching (MCM) score provides a promising solution consiste
    
[^13]: ContactArt：学习类别级联结物体和手部姿态估计的三维交互先验

    ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation. (arXiv:2305.01618v1 [cs.CV])

    [http://arxiv.org/abs/2305.01618](http://arxiv.org/abs/2305.01618)

    本研究提出了一种基于视觉遥操作的数据收集方法以及学习手物互动先验的新方法，从而能够在联结目标和手部姿态估计中实现更好的关键点定位性能。

    

    我们提出了一个新的数据集和一种新方法，用于学习手部和联结目标姿态估计中的手物互动先验。我们首先使用视觉遥操作收集了一个数据集，其中人类操作员可以直接在物理模拟器中游戏来操纵联结对象。 我们记录数据并从模拟器获得有关目标姿态和接触信息的免费和准确注释。 我们的系统仅需要使用iPhone来记录人手运动，可以轻松扩展并大大降低数据和注释收集的成本。使用这些数据，我们学习了三维交互先验，包括捕获对象部件排列分布的鉴别器（在GAN中），以及生成联结对象上接触区域的扩散模型，以指导手势估计。这些结构和接触先验可以很容易地转移到现实世界数据，几乎没有任何领域差距。通过使用我们的数据和学习的先验，我们的方法显著提高了关键点定位性能。

    We propose a new dataset and a novel approach to learning hand-object interaction priors for hand and articulated object pose estimation. We first collect a dataset using visual teleoperation, where the human operator can directly play within a physical simulator to manipulate the articulated objects. We record the data and obtain free and accurate annotations on object poses and contact information from the simulator. Our system only requires an iPhone to record human hand motion, which can be easily scaled up and largely lower the costs of data and annotation collection. With this data, we learn 3D interaction priors including a discriminator (in a GAN) capturing the distribution of how object parts are arranged, and a diffusion model which generates the contact regions on articulated objects, guiding the hand pose estimation. Such structural and contact priors can easily transfer to real-world data with barely any domain gap. By using our data and learned priors, our method signific
    
[^14]: 可微分高斯化层用于深度生成模型正则化的逆问题

    Differentiable Gaussianization Layers for Inverse Problems Regularized by Deep Generative Models. (arXiv:2112.03860v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2112.03860](http://arxiv.org/abs/2112.03860)

    该论文提出了一种使用可微数据相关层进行重新参数化和高斯化潜在张量的方法，以约束逆问题为获得高保真度的分布内解决方案，有效解决深度生成模型逆问题中潜在张量偏离期望高斯分布的问题。

    

    深度生成模型如GAN、标准化流和扩散模型是逆问题的强大正则化器，可以帮助减小不适定性并获得高质量的结果。然而，在逆推过程中，这些模型的潜在张量可能会从期望的高维标准高斯分布中脱离，特别是在数据噪声和不准确的正向模型存在的情况下，会导致低保真度的解决方案。为解决这个问题，我们提出使用新颖的可微数据相关层重新参数化和高斯化潜在张量，其中使用自定义操作符解决优化问题。这些拟议的层将逆问题约束为获得高保真度的分布内解决方案。我们在三个反演任务（压缩感知MRI、图像去模糊和准确度受限的非线性偏微分方程反演问题“eikonal tomography”）上使用两种典型的深度生成模型进行了验证。

    Deep generative models such as GANs, normalizing flows, and diffusion models are powerful regularizers for inverse problems. They exhibit great potential for helping reduce ill-posedness and attain high-quality results. However, the latent tensors of such deep generative models can fall out of the desired high-dimensional standard Gaussian distribution during inversion, particularly in the presence of data noise and inaccurate forward models, leading to low-fidelity solutions. To address this issue, we propose to reparameterize and Gaussianize the latent tensors using novel differentiable data-dependent layers wherein custom operators are defined by solving optimization problems. These proposed layers constrain inverse problems to obtain high-fidelity in-distribution solutions. We validate our technique on three inversion tasks: compressive-sensing MRI, image deblurring, and eikonal tomography (a nonlinear PDE-constrained inverse problem) using two representative deep generative models
    

