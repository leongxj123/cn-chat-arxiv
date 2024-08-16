# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition](https://arxiv.org/abs/2403.17695) | 改进了视觉识别中的非层次Mamba模型，通过改进连续2D扫描过程和方向感知更新，提高了从二维图像中学习特征的能力。 |
| [^2] | [SM4Depth: Seamless Monocular Metric Depth Estimation across Multiple Cameras and Scenes by One Model](https://arxiv.org/abs/2403.08556) | SM4Depth通过一种新的预处理单元和深度间隔离散化的方法，解决了单目度量深度估计中的相机敏感性、场景精度不一致和数据依赖性等问题。 |
| [^3] | [Label Dropout: Improved Deep Learning Echocardiography Segmentation Using Multiple Datasets With Domain Shift and Partial Labelling](https://arxiv.org/abs/2403.07818) | 本文研究利用多个数据集进行深度学习超声心动图分割，在处理部分标记数据时采用改进的交叉熵损失函数。 |
| [^4] | [Active Generation for Image Classification](https://arxiv.org/abs/2403.06517) | 该论文提出了一种名为ActGen的方法，通过采用训练感知的方式来生成图像，以提高图像分类准确性。ActGen利用主动学习的理念，生成类似于挑战性或被误分类样本的图像，并将其整合到训练集中，从而增强模型性能。 |
| [^5] | [WATonoBus: An All Weather Autonomous Shuttle](https://arxiv.org/abs/2312.00938) | 提出了一种考虑恶劣天气的多模块和模块化系统架构，在WATonoBus平台上进行了实际测试，证明其能够解决全天候自动驾驶车辆面临的挑战 |
| [^6] | [Challenging Common Paradigms in Multi-Task Learning](https://arxiv.org/abs/2311.04698) | 我们挑战了多任务学习中的常见范式，通过研究在单任务学习中的影响，揭示了优化器选择在MTL中的关键作用，并理论推导出了梯度冲突的角色。 |
| [^7] | [Long-Tailed Classification Based on Coarse-Grained Leading Forest and Multi-Center Loss.](http://arxiv.org/abs/2310.08206) | 本论文提出了一种基于粗粒度引导森林和多中心损失的长尾分类框架，名为Cognisance。该框架致力于解决长尾分类问题中的类间和类内不平衡，并通过不变特征学习构建多粒度联合解决模型。 |
| [^8] | [End-to-end Autonomous Driving: Challenges and Frontiers.](http://arxiv.org/abs/2306.16927) | 这项研究调查了端到端自动驾驶领域中的关键挑战和未来趋势，包括多模态、可解释性、因果混淆、鲁棒性和世界模型等。通过联合特征优化感知和规划，端到端系统在感知和规划上获得了更好的效果。 |
| [^9] | [Controllable Motion Diffusion Model.](http://arxiv.org/abs/2306.00416) | 该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。 |

# 详细

[^1]: PlainMamba：改进视觉识别中的非层次Mamba

    PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition

    [https://arxiv.org/abs/2403.17695](https://arxiv.org/abs/2403.17695)

    改进了视觉识别中的非层次Mamba模型，通过改进连续2D扫描过程和方向感知更新，提高了从二维图像中学习特征的能力。

    

    我们提出PlainMamba：一种简单的非层次状态空间模型（SSM），旨在用于一般的视觉识别。最近的Mamba模型展示了如何在顺序数据上SSM可以与其他架构竞争激烈，并已初步尝试将其应用于图像。在本文中，我们进一步改进了Mamba的选择性扫描过程以适应视觉领域，通过（i）通过确保在扫描序列中令牌相邻来改善空间连续性的连续2D扫描过程，以及（ii）启用模型区分令牌的空间关系的方向感知更新，通过编码方向信息。我们的架构设计易于使用和易于扩展，由堆叠相同的PlainMamba块形成，结果是始终具有恒定宽度的模型。通过去除

    arXiv:2403.17695v1 Announce Type: cross  Abstract: We present PlainMamba: a simple non-hierarchical state space model (SSM) designed for general visual recognition. The recent Mamba model has shown how SSMs can be highly competitive with other architectures on sequential data and initial attempts have been made to apply it to images. In this paper, we further adapt the selective scanning process of Mamba to the visual domain, enhancing its ability to learn features from two-dimensional images by (i) a continuous 2D scanning process that improves spatial continuity by ensuring adjacency of tokens in the scanning sequence, and (ii) direction-aware updating which enables the model to discern the spatial relations of tokens by encoding directional information. Our architecture is designed to be easy to use and easy to scale, formed by stacking identical PlainMamba blocks, resulting in a model with constant width throughout all layers. The architecture is further simplified by removing the 
    
[^2]: SM4Depth: 一种通过单一模型实现跨多摄像头和场景的无缝单目度量深度估计

    SM4Depth: Seamless Monocular Metric Depth Estimation across Multiple Cameras and Scenes by One Model

    [https://arxiv.org/abs/2403.08556](https://arxiv.org/abs/2403.08556)

    SM4Depth通过一种新的预处理单元和深度间隔离散化的方法，解决了单目度量深度估计中的相机敏感性、场景精度不一致和数据依赖性等问题。

    

    单目度量深度估计（MMDE）的泛化一直是一个长期存在的挑战。最近的方法通过结合相对深度和度量深度或对齐输入图像焦距取得了进展。然而，它们仍然面临着在相机、场景和数据级别上的挑战：（1）对不同摄像头的敏感性；（2）在不同场景中精度不一致；（3）依赖大规模训练数据。本文提出了一种无缝的MMDE方法SM4Depth，以在单个网络内解决上述所有问题。

    arXiv:2403.08556v1 Announce Type: cross  Abstract: The generalization of monocular metric depth estimation (MMDE) has been a longstanding challenge. Recent methods made progress by combining relative and metric depth or aligning input image focal length. However, they are still beset by challenges in camera, scene, and data levels: (1) Sensitivity to different cameras; (2) Inconsistent accuracy across scenes; (3) Reliance on massive training data. This paper proposes SM4Depth, a seamless MMDE method, to address all the issues above within a single network. First, we reveal that a consistent field of view (FOV) is the key to resolve ``metric ambiguity'' across cameras, which guides us to propose a more straightforward preprocessing unit. Second, to achieve consistently high accuracy across scenes, we explicitly model the metric scale determination as discretizing the depth interval into bins and propose variation-based unnormalized depth bins. This method bridges the depth gap of divers
    
[^3]: 标签丢失率：利用具有域转移和部分标记的多个数据集改进深度学习超声心动图分割

    Label Dropout: Improved Deep Learning Echocardiography Segmentation Using Multiple Datasets With Domain Shift and Partial Labelling

    [https://arxiv.org/abs/2403.07818](https://arxiv.org/abs/2403.07818)

    本文研究利用多个数据集进行深度学习超声心动图分割，在处理部分标记数据时采用改进的交叉熵损失函数。

    

    超声心动图（超声）是评估心脏功能时使用的第一种成像方式。从超声中测量功能生物标志物依赖于对心脏结构进行分割，深度学习模型被提出来自动化这一过程。然而，为了将这些工具转化为广泛的临床应用，重要的是分割模型对各种图像具有鲁棒性（例如，由不同扫描仪获得，由不同级别的专家操作员获得等）。为了实现这种鲁棒性水平，有必要使用多个不同的数据集来训练模型。在使用多个不同的数据集进行训练时面临的一个重要挑战是标签存在的变化，即合并数据通常是部分标记的。已经提出了交叉熵损失函数的改进来处理部分标记数据。在本文中，我们展示了训练的naively

    arXiv:2403.07818v1 Announce Type: cross  Abstract: Echocardiography (echo) is the first imaging modality used when assessing cardiac function. The measurement of functional biomarkers from echo relies upon the segmentation of cardiac structures and deep learning models have been proposed to automate the segmentation process. However, in order to translate these tools to widespread clinical use it is important that the segmentation models are robust to a wide variety of images (e.g. acquired from different scanners, by operators with different levels of expertise etc.). To achieve this level of robustness it is necessary that the models are trained with multiple diverse datasets. A significant challenge faced when training with multiple diverse datasets is the variation in label presence, i.e. the combined data are often partially-labelled. Adaptations of the cross entropy loss function have been proposed to deal with partially labelled data. In this paper we show that training naively 
    
[^4]: 图像分类的主动生成

    Active Generation for Image Classification

    [https://arxiv.org/abs/2403.06517](https://arxiv.org/abs/2403.06517)

    该论文提出了一种名为ActGen的方法，通过采用训练感知的方式来生成图像，以提高图像分类准确性。ActGen利用主动学习的理念，生成类似于挑战性或被误分类样本的图像，并将其整合到训练集中，从而增强模型性能。

    

    最近，深度生成模型不断增强的能力突显了它们在提高图像分类准确性方面的潜力。然而，现有方法往往要求生成的图像数量远远超过原始数据集，而在准确性方面只有极小的改进。这种计算昂贵且耗时的过程阻碍了这种方法的实用性。在本文中，我们提出通过专注于模型的具体需求和特征来提高图像生成的效率。我们的方法ActGen以主动学习为中心原则，采用了一个针对训练感知的图像生成方法。它旨在创建类似于当前模型遇到的具有挑战性或被误分类样本的图像，并将这些生成的图像纳入训练集以增强模型性能。

    arXiv:2403.06517v1 Announce Type: cross  Abstract: Recently, the growing capabilities of deep generative models have underscored their potential in enhancing image classification accuracy. However, existing methods often demand the generation of a disproportionately large number of images compared to the original dataset, while having only marginal improvements in accuracy. This computationally expensive and time-consuming process hampers the practicality of such approaches. In this paper, we propose to address the efficiency of image generation by focusing on the specific needs and characteristics of the model. With a central tenet of active learning, our method, named ActGen, takes a training-aware approach to image generation. It aims to create images akin to the challenging or misclassified samples encountered by the current model and incorporates these generated images into the training set to augment model performance. ActGen introduces an attentive image guidance technique, usin
    
[^5]: WATonoBus：一种全天候自动巡航车

    WATonoBus: An All Weather Autonomous Shuttle

    [https://arxiv.org/abs/2312.00938](https://arxiv.org/abs/2312.00938)

    提出了一种考虑恶劣天气的多模块和模块化系统架构，在WATonoBus平台上进行了实际测试，证明其能够解决全天候自动驾驶车辆面临的挑战

    

    自动驾驶车辆在全天候运行中面临显著挑战，涵盖了从感知和决策到路径规划和控制的各个模块。复杂性源于需要解决像雨、雪和雾等恶劣天气条件在自主性堆栈中的问题。传统的基于模型和单模块方法通常缺乏与上游或下游任务的整体集成。我们通过提出一个考虑恶劣天气的多模块和模块化系统架构来解决这个问题，涵盖了从感知水平到决策和安全监测的各个方面，例如覆盖雪的路缘检测。通过在WATonoBus平台上每周日常服务近一年，我们展示了我们提出的方法能够解决恶劣天气条件，并从运营中观察到的极端情况中获得宝贵的经验教训。

    arXiv:2312.00938v1 Announce Type: cross  Abstract: Autonomous vehicle all-weather operation poses significant challenges, encompassing modules from perception and decision-making to path planning and control. The complexity arises from the need to address adverse weather conditions like rain, snow, and fog across the autonomy stack. Conventional model-based and single-module approaches often lack holistic integration with upstream or downstream tasks. We tackle this problem by proposing a multi-module and modular system architecture with considerations for adverse weather across the perception level, through features such as snow covered curb detection, to decision-making and safety monitoring. Through daily weekday service on the WATonoBus platform for almost a year, we demonstrate that our proposed approach is capable of addressing adverse weather conditions and provide valuable learning from edge cases observed during operation.
    
[^6]: 在多任务学习中挑战常见范式

    Challenging Common Paradigms in Multi-Task Learning

    [https://arxiv.org/abs/2311.04698](https://arxiv.org/abs/2311.04698)

    我们挑战了多任务学习中的常见范式，通过研究在单任务学习中的影响，揭示了优化器选择在MTL中的关键作用，并理论推导出了梯度冲突的角色。

    

    尽管近年来多任务学习（MTL）受到了极大关注，但其基本机制仍然知之甚少。最近的方法并未带来一致的性能改进，相比单任务学习（STL）基线，强调了更深入了解MTL特定挑战的重要性。在我们的研究中，我们挑战了MTL中的范式，提出了几点关于STL的重要影响：首先，优化器的选择对MTL的影响只受到了轻微的调查。我们通过各种实验的实证方法展示了常见STL工具（例如Adam优化器）在MTL中的关键作用。为了进一步研究Adam的有效性，我们在一定的假设下从理论上推导出部分损失尺度不变性。其次，梯度冲突的概念经常被描述为MTL中的一个特定问题。我们深入探讨了梯度冲突在MTL中的作用，并将其与STL进行比较。在角度梯度对齐方面，我们没有找到

    arXiv:2311.04698v3 Announce Type: replace-cross  Abstract: While multi-task learning (MTL) has gained significant attention in recent years, its underlying mechanisms remain poorly understood. Recent methods did not yield consistent performance improvements over single task learning (STL) baselines, underscoring the importance of gaining more profound insights about challenges specific to MTL. In our study, we challenge paradigms in MTL in the context of STL: First, the impact of the choice of optimizer has only been mildly investigated in MTL. We show the pivotal role of common STL tools such as the Adam optimizer in MTL empirically in various experiments. To further investigate Adam's effectiveness, we theoretical derive a partial loss-scale invariance under mild assumptions. Second, the notion of gradient conflicts has often been phrased as a specific problem in MTL. We delve into the role of gradient conflicts in MTL and compare it to STL. For angular gradient alignment we find no 
    
[^7]: 基于粗粒度引导森林和多中心损失的长尾分类

    Long-Tailed Classification Based on Coarse-Grained Leading Forest and Multi-Center Loss. (arXiv:2310.08206v1 [cs.CV])

    [http://arxiv.org/abs/2310.08206](http://arxiv.org/abs/2310.08206)

    本论文提出了一种基于粗粒度引导森林和多中心损失的长尾分类框架，名为Cognisance。该框架致力于解决长尾分类问题中的类间和类内不平衡，并通过不变特征学习构建多粒度联合解决模型。

    

    长尾分类是现实世界中不可避免且具有挑战性的问题。大部分现有的长尾分类方法仅关注解决类间不平衡，即头部类别的样本比尾部类别的样本多，而忽略了类内不平衡，即同一类别中头部属性样本数量远大于尾部属性样本数量。模型的偏差是由这两个因素引起的，由于大多数数据集中的属性是隐含的且属性组合非常复杂，处理类内不平衡更加困难。为此，我们提出了一种基于粗粒度引导森林（CLF）和多中心损失（MCL）的长尾分类框架，名为Cognisance，旨在通过不变特征学习构建多粒度联合解决模型。在这个方法中，我们设计了一种新颖的样本选择策略和损失函数，以平衡不同类别和属性之间的样本分布。

    Long-tailed(LT) classification is an unavoidable and challenging problem in the real world. Most of the existing long-tailed classification methods focus only on solving the inter-class imbalance in which there are more samples in the head class than in the tail class, while ignoring the intra-lass imbalance in which the number of samples of the head attribute within the same class is much larger than the number of samples of the tail attribute. The deviation in the model is caused by both of these factors, and due to the fact that attributes are implicit in most datasets and the combination of attributes is very complex, the intra-class imbalance is more difficult to handle. For this purpose, we proposed a long-tailed classification framework, known as \textbf{\textsc{Cognisance}}, which is founded on Coarse-Grained Leading Forest (CLF) and Multi-Center Loss (MCL), aiming to build a multi-granularity joint solution model by means of invariant feature learning. In this method, we desig
    
[^8]: 线束自动驾驶：挑战与前景

    End-to-end Autonomous Driving: Challenges and Frontiers. (arXiv:2306.16927v1 [cs.RO])

    [http://arxiv.org/abs/2306.16927](http://arxiv.org/abs/2306.16927)

    这项研究调查了端到端自动驾驶领域中的关键挑战和未来趋势，包括多模态、可解释性、因果混淆、鲁棒性和世界模型等。通过联合特征优化感知和规划，端到端系统在感知和规划上获得了更好的效果。

    

    自动驾驶领域正在迅速发展，越来越多的方法采用端到端算法框架，利用原始传感器输入生成车辆运动计划，而不是专注于诸如检测和运动预测等单个任务。与模块化流水线相比，端到端系统通过联合特征优化感知和规划来获益。这一领域因大规模数据集的可用性、闭环评估以及自动驾驶算法在挑战性场景中的有效执行所需的需求而蓬勃发展。在本调查中，我们全面分析了250多篇论文，涵盖了端到端自动驾驶的动机、路线图、方法论、挑战和未来趋势。我们深入探讨了多模态、可解释性、因果混淆、鲁棒性和世界模型等几个关键挑战。此外，我们还讨论了基础技术的最新进展。

    The autonomous driving community has witnessed a rapid growth in approaches that embrace an end-to-end algorithm framework, utilizing raw sensor input to generate vehicle motion plans, instead of concentrating on individual tasks such as detection and motion prediction. End-to-end systems, in comparison to modular pipelines, benefit from joint feature optimization for perception and planning. This field has flourished due to the availability of large-scale datasets, closed-loop evaluation, and the increasing need for autonomous driving algorithms to perform effectively in challenging scenarios. In this survey, we provide a comprehensive analysis of more than 250 papers, covering the motivation, roadmap, methodology, challenges, and future trends in end-to-end autonomous driving. We delve into several critical challenges, including multi-modality, interpretability, causal confusion, robustness, and world models, amongst others. Additionally, we discuss current advancements in foundation
    
[^9]: 可控运动扩散模型

    Controllable Motion Diffusion Model. (arXiv:2306.00416v1 [cs.CV])

    [http://arxiv.org/abs/2306.00416](http://arxiv.org/abs/2306.00416)

    该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。

    

    在计算机动画中，为虚拟角色生成逼真且可控的运动是一项具有挑战性的任务。最近的研究从图像生成的扩散模型的成功中汲取灵感，展示了解决这个问题的潜力。然而，这些研究大多限于离线应用，目标是生成同时生成所有步骤的序列级生成。为了能够在响应于时变控制信号的情况下使用扩散模型实现实时运动合成，我们提出了可控运动扩散模型（COMODO）框架。我们的框架以自回归运动扩散模型（A-MDM）为基础，逐步生成运动序列。通过简单地使用标准DDPM算法而无需任何额外复杂性，我们的框架能够产生在不同类型的运动控制下长时间内的高保真度运动序列。

    Generating realistic and controllable motions for virtual characters is a challenging task in computer animation, and its implications extend to games, simulations, and virtual reality. Recent studies have drawn inspiration from the success of diffusion models in image generation, demonstrating the potential for addressing this task. However, the majority of these studies have been limited to offline applications that target at sequence-level generation that generates all steps simultaneously. To enable real-time motion synthesis with diffusion models in response to time-varying control signals, we propose the framework of the Controllable Motion Diffusion Model (COMODO). Our framework begins with an auto-regressive motion diffusion model (A-MDM), which generates motion sequences step by step. In this way, simply using the standard DDPM algorithm without any additional complexity, our framework is able to generate high-fidelity motion sequences over extended periods with different type
    

