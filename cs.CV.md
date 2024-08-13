# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tracking-Assisted Object Detection with Event Cameras](https://arxiv.org/abs/2403.18330) | 本文利用跟踪策略和自动标记算法，针对事件相机中的难以辨识的对象，揭示其特征信息。 |
| [^2] | [Confidence Self-Calibration for Multi-Label Class-Incremental Learning](https://arxiv.org/abs/2403.12559) | 本文提出了一种针对多标签类增量学习中的信心自校准问题的方法，通过引入图卷积网络和最大熵正则化，改善了多标签信心校准，减少了过度自信的误报错误。 |
| [^3] | [CT evaluation of 2D and 3D holistic deep learning methods for the volumetric segmentation of airway lesions](https://arxiv.org/abs/2403.08042) | 本研究比较了2D和3D格式的卷积神经网络在气道病变体积分割方面的能力，发现3D模型在捕捉复杂特征方面表现更优异，并通过对2D模型实施细微结构分割损失来提高准确性，通过外部验证证实了研究结果的稳健性 |
| [^4] | [FSL Model can Score Higher as It Is](https://arxiv.org/abs/2402.18292) | 为了增加测试期间正确预测的机会，研究旨在通过图像到图像的转换纠正FSL模型的测试输入，生成被测试类别的新样本。 |
| [^5] | [Toward a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning](https://arxiv.org/abs/2311.17693) | 利用模拟图像引导的以外科医生为中心的自主机器人学徒系统，通过强化学习和模仿学习代理在眼科白内障手术中适应外科医生的技能水平和外科手术偏好。 |
| [^6] | [EdgeOL: Efficient in-situ Online Learning on Edge Devices.](http://arxiv.org/abs/2401.16694) | 本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。 |
| [^7] | [FedRSU: Federated Learning for Scene Flow Estimation on Roadside Units.](http://arxiv.org/abs/2401.12862) | FedRSU是一种创新的联邦学习框架，用于自监督场景流估计，它通过整合RSU收集的数据，采用循环自监督训练范式来提高模型的准确性和稳定性。 |
| [^8] | [RAVEN: Rethinking Adversarial Video Generation with Efficient Tri-plane Networks.](http://arxiv.org/abs/2401.06035) | 这项研究提出了一种新的无条件视频生成模型，通过高效的三平面网络以及联合帧建模方法和基于光流的模块，实现了高效、时间连贯且无视觉伪影的视频生成。 |
| [^9] | [CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration.](http://arxiv.org/abs/2309.14660) | CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。 |
| [^10] | [LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds.](http://arxiv.org/abs/2308.09908) | 本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。 |
| [^11] | [Towards Trustworthy Dataset Distillation.](http://arxiv.org/abs/2307.09165) | 本论文提出了一种名为可信赖的数据集精炼（TrustDD）的新范式，通过同时考虑内部分布（InD）分类和外部分布（OOD）检测的问题，将大型数据集精炼为小型合成数据集，从而提高模型的效率和可信赖性。 |
| [^12] | [Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models.](http://arxiv.org/abs/2305.13840) | 这篇论文提出了一种基于控制信号的可控文本生成视频的模型，通过空间-时间自注意机制和残差噪声初始化策略，可以生成更连贯的超高质量视频，成功实现了资源高效的收敛。 |
| [^13] | [A Billion-scale Foundation Model for Remote Sensing Images.](http://arxiv.org/abs/2304.05215) | 本文介绍了一个用于遥感图像的十亿级基础模型，并研究了增加模型参数数量对该模型在下游任务中的性能影响，实验显示增加模型参数数量可以显著提高性能。 |

# 详细

[^1]: 使用跟踪辅助的事件相机目标检测

    Tracking-Assisted Object Detection with Event Cameras

    [https://arxiv.org/abs/2403.18330](https://arxiv.org/abs/2403.18330)

    本文利用跟踪策略和自动标记算法，针对事件相机中的难以辨识的对象，揭示其特征信息。

    

    最近，由于事件相机具有高动态范围和无动态模糊等特殊属性，事件驱动目标检测在计算机视觉领域引起了关注。然而，由于特征的不同步性和稀疏性导致了由于相机与之没有相对运动而导致的看不见的对象，这对任务构成了重大挑战。本文将这些看不见的对象视为伪遮挡对象，并旨在揭示其特征。首先，我们引入了对象的可见性属性，并提出了一个自动标记算法，用于在现有的事件相机数据集上附加额外的可见性标签。其次，我们利用跟踪策略来

    arXiv:2403.18330v1 Announce Type: cross  Abstract: Event-based object detection has recently garnered attention in the computer vision community due to the exceptional properties of event cameras, such as high dynamic range and no motion blur. However, feature asynchronism and sparsity cause invisible objects due to no relative motion to the camera, posing a significant challenge in the task. Prior works have studied various memory mechanisms to preserve as many features as possible at the current time, guided by temporal clues. While these implicit-learned memories retain some short-term information, they still struggle to preserve long-term features effectively. In this paper, we consider those invisible objects as pseudo-occluded objects and aim to reveal their features. Firstly, we introduce visibility attribute of objects and contribute an auto-labeling algorithm to append additional visibility labels on an existing event camera dataset. Secondly, we exploit tracking strategies fo
    
[^2]: 多标签类增量学习的信心自校准

    Confidence Self-Calibration for Multi-Label Class-Incremental Learning

    [https://arxiv.org/abs/2403.12559](https://arxiv.org/abs/2403.12559)

    本文提出了一种针对多标签类增量学习中的信心自校准问题的方法，通过引入图卷积网络和最大熵正则化，改善了多标签信心校准，减少了过度自信的误报错误。

    

    多标签类增量学习（MLCIL）中的部分标签挑战是在训练期间只有新类别被标记，而过去和未来标签仍然不可用。这个问题会导致由于错误地高置信度多标签预测而出现大量误报错误，加剧了在不同标签空间内的灾难性遗忘。在本文中，我们旨在在MLCIL中改进多标签信心校准，并提出了一种信心自校准（CSC）方法。首先，为了标签关系校准，我们引入一个类增量图卷积网络，通过构建可学习的、动态扩展的标签关系图来连接孤立的标签空间。然后，为了信心校准，我们针对每个多标签增量提出了一种最大熵正则化，通过对过于自信的输出分布进行惩罚，促进了信心的自校准。

    arXiv:2403.12559v1 Announce Type: cross  Abstract: The partial label challenge in Multi-Label Class-Incremental Learning (MLCIL) arises when only the new classes are labeled during training, while past and future labels remain unavailable. This issue leads to a proliferation of false-positive errors due to erroneously high confidence multi-label predictions, exacerbating catastrophic forgetting within the disjoint label space. In this paper, we aim to refine multi-label confidence calibration in MLCIL and propose a Confidence Self-Calibration (CSC) approach. Firstly, for label relationship calibration, we introduce a class-incremental graph convolutional network that bridges the isolated label spaces by constructing learnable, dynamically extended label relationship graph. Then, for confidence calibration, we present a max-entropy regularization for each multi-label increment, facilitating confidence self-calibration through the penalization of over-confident output distributions. Our 
    
[^3]: CT评估2D和3D整体深度学习方法用于气道病变的体积分割

    CT evaluation of 2D and 3D holistic deep learning methods for the volumetric segmentation of airway lesions

    [https://arxiv.org/abs/2403.08042](https://arxiv.org/abs/2403.08042)

    本研究比较了2D和3D格式的卷积神经网络在气道病变体积分割方面的能力，发现3D模型在捕捉复杂特征方面表现更优异，并通过对2D模型实施细微结构分割损失来提高准确性，通过外部验证证实了研究结果的稳健性

    

    这项研究对卷积神经网络（CNNs）在2D和3D格式中的整体分割能力进行了比较探讨，重点关注囊性纤维化（CF）病变。研究利用了来自两个CF参考中心的数据，涵盖了五个主要的CF结构变化。首先比较了2D和3D模型，突出了3D模型在捕捉粘液栓和实变等复杂特征方面的优越能力。为了提高2D模型的性能，实施和评估了一种适用于细微结构分割的损失，显著提高了其准确性，尽管没有超越3D模型的性能。模型经过进一步通过对肺功能测试（PFTs）的外部评估进行验证，确认了研究结果的稳健性。此外，这项研究不仅限于比较指标；还包括对模型解释性的全面评估。

    arXiv:2403.08042v1 Announce Type: cross  Abstract: This research embarked on a comparative exploration of the holistic segmentation capabilities of Convolutional Neural Networks (CNNs) in both 2D and 3D formats, focusing on cystic fibrosis (CF) lesions. The study utilized data from two CF reference centers, covering five major CF structural changes. Initially, it compared the 2D and 3D models, highlighting the 3D model's superior capability in capturing complex features like mucus plugs and consolidations. To improve the 2D model's performance, a loss adapted to fine structures segmentation was implemented and evaluated, significantly enhancing its accuracy, though not surpassing the 3D model's performance. The models underwent further validation through external evaluation against pulmonary function tests (PFTs), confirming the robustness of the findings. Moreover, this study went beyond comparing metrics; it also included comprehensive assessments of the models' interpretability and 
    
[^4]: FSL模型可以因为其优越性得分更高

    FSL Model can Score Higher as It Is

    [https://arxiv.org/abs/2402.18292](https://arxiv.org/abs/2402.18292)

    为了增加测试期间正确预测的机会，研究旨在通过图像到图像的转换纠正FSL模型的测试输入，生成被测试类别的新样本。

    

    在日常生活中，为了增加被正确识别的机会，我们倾向于面对面地直视面部识别机，而不是侧着面对。少样本学习（FSL）分类本身就具有挑战性，因为模型必须识别属于训练时未见的类别的图像。因此，在测试期间对扭曲和非典型的查询或支持图像会让模型更难正确预测。在我们的研究中，为了增加测试期间正确预测的机会，我们旨在通过图像到图像的转换纠正训练过的FSL模型的测试输入，生成被测试类别的新样本。FSL模型通常是在具有足够样本的类别上进行训练，然后在具有少样本样本的类别上进行测试。我们提出的方法首先捕捉测试图像的风格或形状，然后识别一个适当的训

    arXiv:2402.18292v1 Announce Type: cross  Abstract: In daily life, we tend to present the front of our faces by staring squarely at a facial recognition machine, instead of facing it sideways, in order to increase the chance of being correctly recognised. Few-shot-learning (FSL) classification is challenging in itself because a model has to identify images that belong to classes previously unseen during training. Therefore, a warped and non-typical query or support image during testing can make it even more challenging for a model to predict correctly. In our work, to increase the chance of correct prediction during testing, we aim to rectify the test input of a trained FSL model by generating new samples of the tested classes through image-to-image translation. An FSL model is usually trained on classes with sufficient samples, and then tested on classes with few-shot samples. Our proposed method first captures the style or shape of the test image, and then identifies a suitable traine
    
[^5]: 利用强化学习和模仿学习的外科医生参与眼科机器人学徒系统研究

    Toward a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2311.17693](https://arxiv.org/abs/2311.17693)

    利用模拟图像引导的以外科医生为中心的自主机器人学徒系统，通过强化学习和模仿学习代理在眼科白内障手术中适应外科医生的技能水平和外科手术偏好。

    

    机器人辅助手术系统在提高手术精确度和减少人为错误方面展示了显著潜力。然而，现有系统缺乏适应个别外科医生的独特偏好和要求的能力。此外，它们主要集中在普通手术（如腹腔镜手术），不适用于非常精密的微创手术，如眼科手术。因此，我们提出了一种基于模拟图像引导的以外科医生为中心的自主机器人学徒系统，可在眼科白内障手术过程中适应个别外科医生的技能水平和首选外科手术技术。我们的方法利用模拟环境来训练以图像数据为指导的强化学习和模仿学习代理，以执行白内障手术的切口阶段所有任务。通过将外科医生的动作和偏好整合到训练过程中，让外科医生参与其中，我们的方法可以达到更好的效果。

    arXiv:2311.17693v2 Announce Type: replace-cross  Abstract: Robotic-assisted surgical systems have demonstrated significant potential in enhancing surgical precision and minimizing human errors. However, existing systems lack the ability to accommodate the unique preferences and requirements of individual surgeons. Additionally, they primarily focus on general surgeries (e.g., laparoscopy) and are not suitable for highly precise microsurgeries, such as ophthalmic procedures. Thus, we propose a simulation-based image-guided approach for surgeon-centered autonomous agents that can adapt to the individual surgeon's skill level and preferred surgical techniques during ophthalmic cataract surgery. Our approach utilizes a simulated environment to train reinforcement and imitation learning agents guided by image data to perform all tasks of the incision phase of cataract surgery. By integrating the surgeon's actions and preferences into the training process with the surgeon-in-the-loop, our ap
    
[^6]: EdgeOL: 边缘设备上高效的原位在线学习

    EdgeOL: Efficient in-situ Online Learning on Edge Devices. (arXiv:2401.16694v1 [cs.LG])

    [http://arxiv.org/abs/2401.16694](http://arxiv.org/abs/2401.16694)

    本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。

    

    新兴应用，如机器人辅助养老和物体识别，通常采用深度学习神经网络模型，并且自然需要：i) 处理实时推理请求和ii) 适应可能的部署场景变化。在线模型微调被广泛采用以满足这些需求。然而，微调会导致显著的能量消耗，使其难以部署在边缘设备上。在本文中，我们提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率。实验结果显示，EdgeOL平均减少了82%的微调执行时间，74%的能量消耗，并提高了平均推理准确率1.70%，相对于即时在线学习策略。

    Emerging applications, such as robot-assisted eldercare and object recognition, generally employ deep learning neural networks (DNNs) models and naturally require: i) handling streaming-in inference requests and ii) adapting to possible deployment scenario changes. Online model fine-tuning is widely adopted to satisfy these needs. However, fine-tuning involves significant energy consumption, making it challenging to deploy on edge devices. In this paper, we propose EdgeOL, an edge online learning framework that optimizes inference accuracy, fine-tuning execution time, and energy efficiency through both inter-tuning and intra-tuning optimizations. Experimental results show that, on average, EdgeOL reduces overall fine-tuning execution time by 82%, energy consumption by 74%, and improves average inference accuracy by 1.70% over the immediate online learning strategy.
    
[^7]: FedRSU: 基于RSU的联邦学习在场景流估计中的应用

    FedRSU: Federated Learning for Scene Flow Estimation on Roadside Units. (arXiv:2401.12862v1 [cs.CV])

    [http://arxiv.org/abs/2401.12862](http://arxiv.org/abs/2401.12862)

    FedRSU是一种创新的联邦学习框架，用于自监督场景流估计，它通过整合RSU收集的数据，采用循环自监督训练范式来提高模型的准确性和稳定性。

    

    RSU（路边单元）通过车联网（V2X）通信可以显著提高自动驾驶汽车的安全性和稳健性。目前，单个RSU的使用主要集中在实时推断和V2X协作上，而忽视了RSU传感器收集的高质量数据的潜在价值。整合大量来自多个RSU的数据可以为模型训练提供丰富的数据来源。然而，缺乏地面真值标注和传输庞大数据量的困难是充分利用这些隐藏价值的两个不可避免的障碍。在本文中，我们介绍了FedRSU，一种创新的自监督场景流估计的联邦学习框架。在FedRSU中，我们提出了一种循环自监督训练范式，对于每个RSU，可以通过其后续未来的多模态观测对每个时间戳的点的场景流预测进行监督。FedRSU的另一个关键组成部分是联邦

    Roadside unit (RSU) can significantly improve the safety and robustness of autonomous vehicles through Vehicle-to-Everything (V2X) communication. Currently, the usage of a single RSU mainly focuses on real-time inference and V2X collaboration, while neglecting the potential value of the high-quality data collected by RSU sensors. Integrating the vast amounts of data from numerous RSUs can provide a rich source of data for model training. However, the absence of ground truth annotations and the difficulty of transmitting enormous volumes of data are two inevitable barriers to fully exploiting this hidden value. In this paper, we introduce FedRSU, an innovative federated learning framework for self-supervised scene flow estimation. In FedRSU, we present a recurrent self-supervision training paradigm, where for each RSU, the scene flow prediction of points at every timestamp can be supervised by its subsequent future multi-modality observation. Another key component of FedRSU is federated
    
[^8]: RAVEN：用高效的三平面网络重新思考对抗性视频生成

    RAVEN: Rethinking Adversarial Video Generation with Efficient Tri-plane Networks. (arXiv:2401.06035v1 [cs.CV])

    [http://arxiv.org/abs/2401.06035](http://arxiv.org/abs/2401.06035)

    这项研究提出了一种新的无条件视频生成模型，通过高效的三平面网络以及联合帧建模方法和基于光流的模块，实现了高效、时间连贯且无视觉伪影的视频生成。

    

    我们提出了一种新颖的无条件视频生成模型，旨在解决长期的空间和时间依赖性。为了捕捉这些依赖关系，我们的方法将受到三维感知生成框架启发的混合显式-隐式三平面表示法并入，并使用一个唯一的潜在编码来建模整个视频序列。然后，从中间三平面表示法中合成单个视频帧，该表示法本身是从主要潜在编码中派生的。这种新颖的策略将计算复杂度减少了2倍，以FLOPs度量。因此，我们的方法便于高效和时间连贯地生成视频。此外，与自回归方法相比，我们的联合帧建模方法可以减少视觉伪影的产生。我们通过在生成对抗网络（GAN）中集成基于光流的模块进一步增强了模型的功能。

    We present a novel unconditional video generative model designed to address long-term spatial and temporal dependencies. To capture these dependencies, our approach incorporates a hybrid explicit-implicit tri-plane representation inspired by 3D-aware generative frameworks developed for three-dimensional object representation and employs a singular latent code to model an entire video sequence. Individual video frames are then synthesized from an intermediate tri-plane representation, which itself is derived from the primary latent code. This novel strategy reduces computational complexity by a factor of $2$ as measured in FLOPs. Consequently, our approach facilitates the efficient and temporally coherent generation of videos. Moreover, our joint frame modeling approach, in contrast to autoregressive methods, mitigates the generation of visual artifacts. We further enhance the model's capabilities by integrating an optical flow-based module within our Generative Adversarial Network (GAN
    
[^9]: CoFiI2P: 粗到精的图像到点云注册的对应关系

    CoFiI2P: Coarse-to-Fine Correspondences for Image-to-Point Cloud Registration. (arXiv:2309.14660v1 [cs.CV])

    [http://arxiv.org/abs/2309.14660](http://arxiv.org/abs/2309.14660)

    CoFiI2P是一种粗到精的图像到点云注册方法，通过利用全局信息和特征建立对应关系，实现全局最优解。

    

    图像到点云（I2P）注册是机器人导航和移动建图领域中的一项基础任务。现有的I2P注册方法在点到像素级别上估计对应关系，忽略了全局对齐。然而，没有来自全局约束的高级引导的I2P匹配容易收敛到局部最优解。为了解决这个问题，本文提出了一种新的I2P注册网络CoFiI2P，通过粗到精的方式提取对应关系，以得到全局最优解。首先，将图像和点云输入到一个共享编码-解码网络中进行层次化特征提取。然后，设计了一个粗到精的匹配模块，利用特征建立稳健的特征对应关系。具体来说，在粗匹配块中，采用了一种新型的I2P变换模块，从图像和点云中捕捉同质和异质的全局信息。通过判别描述子，完成粗-细特征匹配过程。最后，通过细化匹配模块进一步提升对应关系的准确性。

    Image-to-point cloud (I2P) registration is a fundamental task in the fields of robot navigation and mobile mapping. Existing I2P registration works estimate correspondences at the point-to-pixel level, neglecting the global alignment. However, I2P matching without high-level guidance from global constraints may converge to the local optimum easily. To solve the problem, this paper proposes CoFiI2P, a novel I2P registration network that extracts correspondences in a coarse-to-fine manner for the global optimal solution. First, the image and point cloud are fed into a Siamese encoder-decoder network for hierarchical feature extraction. Then, a coarse-to-fine matching module is designed to exploit features and establish resilient feature correspondences. Specifically, in the coarse matching block, a novel I2P transformer module is employed to capture the homogeneous and heterogeneous global information from image and point cloud. With the discriminate descriptors, coarse super-point-to-su
    
[^10]: LEGO: 对于基于点云的在线多目标跟踪的学习和图优化的模块化跟踪器

    LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds. (arXiv:2308.09908v1 [cs.CV])

    [http://arxiv.org/abs/2308.09908](http://arxiv.org/abs/2308.09908)

    本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。

    

    在线多目标跟踪（MOT）在自主系统中起着关键作用。现有的最先进方法通常采用跟踪-检测方法，数据关联起到了至关重要的作用。本文提出了一个学习和图优化（LEGO）的模块化跟踪器，以提高数据关联性能。所提出的LEGO跟踪器集成了图优化和自注意力机制，能够有效地制定关联评分图，从而实现准确高效的目标匹配。为了进一步增强状态更新过程，本文还添加了卡尔曼滤波器，通过将对象状态的时间连贯性纳入跟踪中，确保一致的跟踪。与其他在线跟踪方法（包括基于LiDAR和基于LiDAR-相机融合的方法）相比，我们提出的仅利用LiDAR的方法表现出了卓越性能。在提交结果至KITTI目标跟踪评估排行榜时，LEGO排名第一。

    Online multi-object tracking (MOT) plays a pivotal role in autonomous systems. The state-of-the-art approaches usually employ a tracking-by-detection method, and data association plays a critical role. This paper proposes a learning and graph-optimized (LEGO) modular tracker to improve data association performance in the existing literature. The proposed LEGO tracker integrates graph optimization and self-attention mechanisms, which efficiently formulate the association score map, facilitating the accurate and efficient matching of objects across time frames. To further enhance the state update process, the Kalman filter is added to ensure consistent tracking by incorporating temporal coherence in the object states. Our proposed method utilizing LiDAR alone has shown exceptional performance compared to other online tracking approaches, including LiDAR-based and LiDAR-camera fusion-based methods. LEGO ranked 1st at the time of submitting results to KITTI object tracking evaluation ranki
    
[^11]: 迈向可信赖的数据集精炼

    Towards Trustworthy Dataset Distillation. (arXiv:2307.09165v1 [cs.LG])

    [http://arxiv.org/abs/2307.09165](http://arxiv.org/abs/2307.09165)

    本论文提出了一种名为可信赖的数据集精炼（TrustDD）的新范式，通过同时考虑内部分布（InD）分类和外部分布（OOD）检测的问题，将大型数据集精炼为小型合成数据集，从而提高模型的效率和可信赖性。

    

    在将深度学习应用于实际应用时，效率和可信赖性是两个永恒的追求。就效率而言，数据集精炼（DD）致力于通过将大型数据集精炼为小型合成数据集来降低训练成本。然而，现有方法仅集中于在封闭世界环境下的内部分布（InD）分类，忽略了外部分布（OOD）样本。另一方面，OOD检测旨在提高模型的可信赖性，在完整数据设置下通常效率低下。我们首次同时考虑了这两个问题，并提出了一种新的范式，称为可信赖的数据集精炼（TrustDD）。通过精炼InD样本和异常值，这些被筛选的数据集能够训练出既擅长InD分类又能进行OOD检测的模型。为了缓解对真实异常值数据的需求，并使OOD检测更加实用，我们进一步提出了对InD样本损坏以生成伪样本的方法。

    Efficiency and trustworthiness are two eternal pursuits when applying deep learning in real-world applications. With regard to efficiency, dataset distillation (DD) endeavors to reduce training costs by distilling the large dataset into a tiny synthetic dataset. However, existing methods merely concentrate on in-distribution (InD) classification in a closed-world setting, disregarding out-of-distribution (OOD) samples. On the other hand, OOD detection aims to enhance models' trustworthiness, which is always inefficiently achieved in full-data settings. For the first time, we simultaneously consider both issues and propose a novel paradigm called Trustworthy Dataset Distillation (TrustDD). By distilling both InD samples and outliers, the condensed datasets are capable to train models competent in both InD classification and OOD detection. To alleviate the requirement of real outlier data and make OOD detection more practical, we further propose to corrupt InD samples to generate pseudo-
    
[^12]: Control-A-Video: 控制性文本生成视频的扩散模型

    Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. (arXiv:2305.13840v1 [cs.CV])

    [http://arxiv.org/abs/2305.13840](http://arxiv.org/abs/2305.13840)

    这篇论文提出了一种基于控制信号的可控文本生成视频的模型，通过空间-时间自注意机制和残差噪声初始化策略，可以生成更连贯的超高质量视频，成功实现了资源高效的收敛。

    

    本文提出了一种基于控制信号的可控文本生成视频（T2V）扩散模型，称为Video-ControlNet。该模型是在预训练的有条件文本生成图像（T2I）扩散模型基础上构建的，其中包括一种空间-时间自注意机制和可训练的时间层，用于有效的跨帧建模。提出了一种第一帧条件策略，以促进模型在自回归方式下生成转换自图像领域以及任意长度视频。此外，Video-ControlNet采用一种基于残差的噪声初始化策略，从输入视频中引入运动先验，从而产生更连贯的视频。通过提出的架构和策略，Video-ControlNet可以实现资源高效的收敛，生成具有细粒度控制的优质一致视频。广泛的实验证明了它的成功。

    This paper presents a controllable text-to-video (T2V) diffusion model, named Video-ControlNet, that generates videos conditioned on a sequence of control signals, such as edge or depth maps. Video-ControlNet is built on a pre-trained conditional text-to-image (T2I) diffusion model by incorporating a spatial-temporal self-attention mechanism and trainable temporal layers for efficient cross-frame modeling. A first-frame conditioning strategy is proposed to facilitate the model to generate videos transferred from the image domain as well as arbitrary-length videos in an auto-regressive manner. Moreover, Video-ControlNet employs a novel residual-based noise initialization strategy to introduce motion prior from an input video, producing more coherent videos. With the proposed architecture and strategies, Video-ControlNet can achieve resource-efficient convergence and generate superior quality and consistent videos with fine-grained control. Extensive experiments demonstrate its success i
    
[^13]: 一种用于遥感图像的十亿级基础模型

    A Billion-scale Foundation Model for Remote Sensing Images. (arXiv:2304.05215v1 [cs.CV])

    [http://arxiv.org/abs/2304.05215](http://arxiv.org/abs/2304.05215)

    本文介绍了一个用于遥感图像的十亿级基础模型，并研究了增加模型参数数量对该模型在下游任务中的性能影响，实验显示增加模型参数数量可以显著提高性能。

    

    随着基础模型在视觉任务中的潜力引起了广泛关注，先对这些模型进行预训练已成为一个关键步骤。预训练基础模型的三个关键因素是预训练方法、预训练数据集的大小以及模型参数的数量。最近，遥感领域的研究主要关注预训练方法和数据集的大小，对模型参数的数量关注较少。本文通过研究增加模型参数数量对基础模型在旋转目标检测和语义分割等下游任务中性能的影响来弥补这一空白。我们使用不同数量参数（包括86M、605.26M、1.3B和2.4B）的基础模型进行预训练，以确定参数增加是否会提高下游任务的性能。据我们所知，这是第一个用于遥感图像的十亿级基础模型。我们的实验表明，增加模型参数数量可以显著提高下游任务的性能。此外，我们还介绍了一个包含10亿个遥感图像的新的预训练数据集，并向研究社区公开。

    As the potential of foundation models in visual tasks has garnered significant attention, pretraining these models before downstream tasks has become a crucial step. The three key factors in pretraining foundation models are the pretraining method, the size of the pretraining dataset, and the number of model parameters. Recently, research in the remote sensing field has focused primarily on the pretraining method and the size of the dataset, with limited emphasis on the number of model parameters. This paper addresses this gap by examining the effect of increasing the number of model parameters on the performance of foundation models in downstream tasks such as rotated object detection and semantic segmentation. We pretrained foundation models with varying numbers of parameters, including 86M, 605.26M, 1.3B, and 2.4B, to determine whether performance in downstream tasks improved with an increase in parameters. To the best of our knowledge, this is the first billion-scale foundation mod
    

