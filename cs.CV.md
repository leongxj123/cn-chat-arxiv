# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery](https://arxiv.org/abs/2404.02877) | 本研究对卫星图像中识别飞机的任务自定义的一套先进对象检测算法进行了全面评估和比较，发现YOLOv5是在不同成像条件下展现高精度和适应性的最优模型。 |
| [^2] | [Rethinking Multi-domain Generalization with A General Learning Objective](https://arxiv.org/abs/2402.18853) | 提出了一个通用学习目标范式，通过Y-mapping来放松约束并设计新的学习目标，包括学习域无关的条件特征和最大化后验概率，通过正则化项解决放松约束引起的问题 |
| [^3] | [Interpretable Semiotics Networks Representing Awareness](https://arxiv.org/abs/2310.05212) | 这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。 |
| [^4] | [Audio-Infused Automatic Image Colorization by Exploiting Audio Scene Semantics.](http://arxiv.org/abs/2401.13270) | 本论文提出了一种使用音频场景语义进行自动图像上色的方法，通过引入音频作为辅助信息，降低了对场景语义理解的难度，并通过三个阶段的网络进行了实现。 |
| [^5] | [Advances in Kidney Biopsy Structural Assessment through Dense Instance Segmentation.](http://arxiv.org/abs/2309.17166) | 这项研究提出了一种基于密集实例分割的肾脏活检结构评估的方法，能够自动统计解剖结构上的统计数据，从而减少工作量和观察者间变异性。 |
| [^6] | [Awesome-META+: Meta-Learning Research and Learning Platform.](http://arxiv.org/abs/2304.12921) | Awesome-META+是一个元学习框架集成和学习平台，旨在提供完整可靠的元学习框架应用和面向初学者的学习材料，进而促进元学习的发展并将其从小众领域转化为主流的研究方向。 |
| [^7] | [No-Box Attacks on 3D Point Cloud Classification.](http://arxiv.org/abs/2210.14164) | 该论文介绍了一种新的方法，可以在不访问目标DNN模型的情况下预测3D点云中的对抗点，提供了无盒子攻击的新视角。 |

# 详细

[^1]: FlightScope: 卫星图像中飞行器检测算法的深度全面评估

    FlightScope: A Deep Comprehensive Assessment of Aircraft Detection Algorithms in Satellite Imagery

    [https://arxiv.org/abs/2404.02877](https://arxiv.org/abs/2404.02877)

    本研究对卫星图像中识别飞机的任务自定义的一套先进对象检测算法进行了全面评估和比较，发现YOLOv5是在不同成像条件下展现高精度和适应性的最优模型。

    

    arXiv:2404.02877v1 公告类型：跨领域 摘要：在遥感卫星图像中进行对象检测对于许多领域，如生物物理学和环境监测至关重要。尽管深度学习算法不断发展，但它们大多在常见的基于地面拍摄的照片上实施和测试。本文对一套针对在卫星图像中识别飞机这一任务定制的先进对象检测算法进行了批判性评估和比较。利用大型HRPlanesV2数据集，以及与GDIT数据集的严格验证，该研究涵盖了一系列方法，包括YOLO版本5和8、Faster RCNN、CenterNet、RetinaNet、RTMDet和DETR，均是从头开始训练的。这项全面的训练和验证研究揭示了YOLOv5作为识别遥感数据中的飞机这一特定案例的卓越模型，展示了其在不同成像条件下的高精度和适应性。

    arXiv:2404.02877v1 Announce Type: cross  Abstract: Object detection in remotely sensed satellite pictures is fundamental in many fields such as biophysical, and environmental monitoring. While deep learning algorithms are constantly evolving, they have been mostly implemented and tested on popular ground-based taken photos. This paper critically evaluates and compares a suite of advanced object detection algorithms customized for the task of identifying aircraft within satellite imagery. Using the large HRPlanesV2 dataset, together with a rigorous validation with the GDIT dataset, this research encompasses an array of methodologies including YOLO versions 5 and 8, Faster RCNN, CenterNet, RetinaNet, RTMDet, and DETR, all trained from scratch. This exhaustive training and validation study reveal YOLOv5 as the preeminent model for the specific case of identifying airplanes from remote sensing data, showcasing high precision and adaptability across diverse imaging conditions. This research
    
[^2]: 重新思考带有通用学习目标的多领域泛化

    Rethinking Multi-domain Generalization with A General Learning Objective

    [https://arxiv.org/abs/2402.18853](https://arxiv.org/abs/2402.18853)

    提出了一个通用学习目标范式，通过Y-mapping来放松约束并设计新的学习目标，包括学习域无关的条件特征和最大化后验概率，通过正则化项解决放松约束引起的问题

    

    多领域泛化（mDG）的普遍目标是最小化训练和测试分布之间的差异，以增强边际到标签分布映射。然而，现有的mDG文献缺乏一个通用的学习目标范式，通常对静态目标边际分布施加约束。在本文中，我们提议利用一个$Y$-mapping来放松约束。我们重新思考了mDG的学习目标，并设计了一个新的通用学习目标来解释和分析大多数现有的mDG智慧。这个通用目标分为两个协同的目标：学习与域无关的条件特征和最大化一个后验。我们探索了两个有效的正则化项，这些项结合了先验信息并抑制了无效的因果关系，减轻了放松约束所带来的问题。我们在理论上为域对齐提供了一个上限。

    arXiv:2402.18853v1 Announce Type: cross  Abstract: Multi-domain generalization (mDG) is universally aimed to minimize the discrepancy between training and testing distributions to enhance marginal-to-label distribution mapping. However, existing mDG literature lacks a general learning objective paradigm and often imposes constraints on static target marginal distributions. In this paper, we propose to leverage a $Y$-mapping to relax the constraint. We rethink the learning objective for mDG and design a new \textbf{general learning objective} to interpret and analyze most existing mDG wisdom. This general objective is bifurcated into two synergistic amis: learning domain-independent conditional features and maximizing a posterior. Explorations also extend to two effective regularization terms that incorporate prior information and suppress invalid causality, alleviating the issues that come with relaxed constraints. We theoretically contribute an upper bound for the domain alignment of 
    
[^3]: 可解释的符号网络代表意识的知觉

    Interpretable Semiotics Networks Representing Awareness

    [https://arxiv.org/abs/2310.05212](https://arxiv.org/abs/2310.05212)

    这个研究描述了一个计算模型，通过追踪和模拟物体感知以及其在交流中所传达的表示来模拟人类的意识。相比于大多数无法解释的神经网络，该模型具有解释性，并可以通过构建新网络来定义物体感知。

    

    人类每天都感知物体，并通过各种渠道传达他们的感知。在这里，我们描述了一个计算模型，追踪和模拟物体的感知以及它们在交流中所传达的表示。我们描述了我们内部表示的两个关键组成部分（"观察到的"和"看到的"），并将它们与熟悉的计算机视觉概念（编码和解码）相关联。这些元素被合并在一起形成符号网络，模拟了物体感知和人类交流中的意识。如今，大多数神经网络都是不可解释的。另一方面，我们的模型克服了这个限制。实验证明了该模型的可见性。我们人的物体感知模型使我们能够通过网络定义物体感知。我们通过构建一个包括基准分类器和额外层的新网络来演示这一点。这个层产生了图像的感知。

    Humans perceive objects daily and communicate their perceptions using various channels. Here, we describe a computational model that tracks and simulates objects' perception and their representations as they are conveyed in communication.   We describe two key components of our internal representation ("observed" and "seen") and relate them to familiar computer vision notions (encoding and decoding). These elements are joined together to form semiotics networks, which simulate awareness in object perception and human communication.   Nowadays, most neural networks are uninterpretable. On the other hand, our model overcomes this limitation. The experiments demonstrates the visibility of the model.   Our model of object perception by a person allows us to define object perception by a network. We demonstrate this with an example of an image baseline classifier by constructing a new network that includes the baseline classifier and an additional layer. This layer produces the images "perc
    
[^4]: 通过利用音频场景语义实现自动图像上色

    Audio-Infused Automatic Image Colorization by Exploiting Audio Scene Semantics. (arXiv:2401.13270v1 [cs.CV])

    [http://arxiv.org/abs/2401.13270](http://arxiv.org/abs/2401.13270)

    本论文提出了一种使用音频场景语义进行自动图像上色的方法，通过引入音频作为辅助信息，降低了对场景语义理解的难度，并通过三个阶段的网络进行了实现。

    

    自动图像上色是一个具有不确定性的问题，需要对场景进行准确的语义理解，以估计灰度图像的合理颜色。虽然最近的基于交互的方法取得了令人印象深刻的性能，但对于自动上色来说，推断出逼真和准确的颜色仍然是一个非常困难的任务。为了降低对灰度场景的语义理解难度，本文尝试利用相应的音频，音频自然地包含了关于同一场景的额外语义信息。具体而言，提出了一种新颖的音频注入自动图像上色（AIAIC）网络，该网络分为三个阶段。首先，我们将彩色图像的语义作为桥梁，通过彩色图像的语义引导预训练上色网络。其次，利用音频与视频的自然共现来学习音频和视觉场景之间的颜色语义相关性。第三，隐式音频语义表示被利用以引导图像上色。

    Automatic image colorization is inherently an ill-posed problem with uncertainty, which requires an accurate semantic understanding of scenes to estimate reasonable colors for grayscale images. Although recent interaction-based methods have achieved impressive performance, it is still a very difficult task to infer realistic and accurate colors for automatic colorization. To reduce the difficulty of semantic understanding of grayscale scenes, this paper tries to utilize corresponding audio, which naturally contains extra semantic information about the same scene. Specifically, a novel audio-infused automatic image colorization (AIAIC) network is proposed, which consists of three stages. First, we take color image semantics as a bridge and pretrain a colorization network guided by color image semantics. Second, the natural co-occurrence of audio and video is utilized to learn the color semantic correlations between audio and visual scenes. Third, the implicit audio semantic representati
    
[^5]: 通过密集实例分割在肾脏活检结构评估方面的进展

    Advances in Kidney Biopsy Structural Assessment through Dense Instance Segmentation. (arXiv:2309.17166v1 [cs.CV])

    [http://arxiv.org/abs/2309.17166](http://arxiv.org/abs/2309.17166)

    这项研究提出了一种基于密集实例分割的肾脏活检结构评估的方法，能够自动统计解剖结构上的统计数据，从而减少工作量和观察者间变异性。

    

    肾脏活检是肾脏疾病诊断的金标准。专家肾脏病理学家制定的病变评分是半定量的，并且存在高的观察者间变异性。因此，通过对分割的解剖对象进行自动统计可以显著减少工作量和这种观察者间变异性。然而，活检的实例分割是一个具有挑战性的问题，原因有：（a）平均数量较大（约300至1000个）密集接触的解剖结构，（b）具有多个类别（至少3个），（c）尺寸和形状各异。目前使用的实例分割模型不能以高效通用的方式同时解决这些挑战。在本文中，我们提出了第一个不需要锚点的实例分割模型，该模型将扩散模型、变换器模块和RCNN（区域卷积神经网络）结合起来。我们的模型在一台NVIDIA GeForce RTX 3090 GPU上进行训练，但可以提供可观的结果。

    The kidney biopsy is the gold standard for the diagnosis of kidney diseases. Lesion scores made by expert renal pathologists are semi-quantitative and suffer from high inter-observer variability. Automatically obtaining statistics per segmented anatomical object, therefore, can bring significant benefits in reducing labor and this inter-observer variability. Instance segmentation for a biopsy, however, has been a challenging problem due to (a) the on average large number (around 300 to 1000) of densely touching anatomical structures, (b) with multiple classes (at least 3) and (c) in different sizes and shapes. The currently used instance segmentation models cannot simultaneously deal with these challenges in an efficient yet generic manner. In this paper, we propose the first anchor-free instance segmentation model that combines diffusion models, transformer modules, and RCNNs (regional convolution neural networks). Our model is trained on just one NVIDIA GeForce RTX 3090 GPU, but can 
    
[^6]: Awesome-META+: 元学习研究与学习平台

    Awesome-META+: Meta-Learning Research and Learning Platform. (arXiv:2304.12921v1 [cs.LG])

    [http://arxiv.org/abs/2304.12921](http://arxiv.org/abs/2304.12921)

    Awesome-META+是一个元学习框架集成和学习平台，旨在提供完整可靠的元学习框架应用和面向初学者的学习材料，进而促进元学习的发展并将其从小众领域转化为主流的研究方向。

    

    人工智能已经在经济、产业、教育等各个领域产生了深远的影响，但还存在诸多限制。元学习，也称为“学习如何学习”，为通用人工智能提供了突破目前瓶颈的机会。然而，元学习起步较晚，相比CV、NLP等领域，项目数量较少。每次部署都需要大量的经验去配置环境、调试代码甚至重写，而且框架之间相对孤立。此外，目前针对元学习的专门平台和面向初学者的学习材料相对较少，门槛相对较高。基于此，Awesome-META+提出了一个元学习框架集成和学习平台，旨在解决上述问题并提供完整可靠的元学习框架应用和学习平台。该项目旨在促进元学习的发展，并将其从一个小众领域转化为一个主流的研究方向。

    Artificial intelligence technology has already had a profound impact in various fields such as economy, industry, and education, but still limited. Meta-learning, also known as "learning to learn", provides an opportunity for general artificial intelligence, which can break through the current AI bottleneck. However, meta learning started late and there are fewer projects compare with CV, NLP etc. Each deployment requires a lot of experience to configure the environment, debug code or even rewrite, and the frameworks are isolated. Moreover, there are currently few platforms that focus exclusively on meta-learning, or provide learning materials for novices, for which the threshold is relatively high. Based on this, Awesome-META+, a meta-learning framework integration and learning platform is proposed to solve the above problems and provide a complete and reliable meta-learning framework application and learning platform. The project aims to promote the development of meta-learning and t
    
[^7]: 3D点云分类的无盒子攻击

    No-Box Attacks on 3D Point Cloud Classification. (arXiv:2210.14164v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.14164](http://arxiv.org/abs/2210.14164)

    该论文介绍了一种新的方法，可以在不访问目标DNN模型的情况下预测3D点云中的对抗点，提供了无盒子攻击的新视角。

    

    对于基于深度神经网络（DNN）的各种输入信号的分析，对抗攻击构成了严重挑战。在3D点云的情况下，已经开发出了一些方法来识别在网络决策中起关键作用的点，而这些方法在生成现有的对抗攻击中变得至关重要。例如，显著性图方法是一种流行的方法，用于识别对抗攻击会显著影响网络决策的点。通常，识别对抗点的方法依赖于对目标DNN模型的访问，以确定哪些点对模型的决策至关重要。本文旨在对这个问题提供一种新的视角，在不访问目标DNN模型的情况下预测对抗点，这被称为“无盒子”攻击。为此，我们定义了14个点云特征，并使用多元线性回归来检查这些特征是否可以用于预测对抗点，以及哪些特征对预测最为重要。

    Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the access to the DNN model itself to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, where adversarial points can be predicted without access to the target DNN model, which is referred to as a ``no-box'' attack. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for adversarial point prediction, and which
    

