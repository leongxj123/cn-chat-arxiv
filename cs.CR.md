# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors](https://rss.arxiv.org/abs/2402.01369) | 本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。 |
| [^2] | [Evaluating Membership Inference Attacks and Defenses in Federated Learning](https://arxiv.org/abs/2402.06289) | 这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。 |
| [^3] | [Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks.](http://arxiv.org/abs/2307.08939) | 这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。 |

# 详细

[^1]: 使用多模式先验的有针对性攻击文本到图像扩散模型

    Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors

    [https://rss.arxiv.org/abs/2402.01369](https://rss.arxiv.org/abs/2402.01369)

    本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。

    

    扩散模型已广泛应用于各种图像生成任务中，展现了图像和文本模态之间的卓越联系。然而，它们面临着被恶意利用的挑战，通过在原始提示后附加特定后缀来生成有害或敏感图像。现有作品主要关注使用单模态信息进行攻击，未能利用多模态特征，导致性能不尽如人意。在本工作中，我们提出了一种名为MMP-Attack的有针对性攻击方法，它将多模态先验（MMP）即文本和图像特征进行整合。具体而言，MMP-Attack的目标是在图像内容中添加目标对象的同时，同时移除原始对象。与现有作品相比，MMP-Attack具有更高的普适性和可转移性，在攻击商业文本到图像（T2I）模型（如DALL-E 3）方面表现出明显优势。据我们所知，这标志着当前最佳的技术水平。

    Diffusion models have been widely deployed in various image generation tasks, demonstrating an extraordinary connection between image and text modalities. However, they face challenges of being maliciously exploited to generate harmful or sensitive images by appending a specific suffix to the original prompt. Existing works mainly focus on using single-modal information to conduct attacks, which fails to utilize multi-modal features and results in less than satisfactory performance. Integrating multi-modal priors (MMP), i.e. both text and image features, we propose a targeted attack method named MMP-Attack in this work. Specifically, the goal of MMP-Attack is to add a target object into the image content while simultaneously removing the original object. The MMP-Attack shows a notable advantage over existing works with superior universality and transferability, which can effectively attack commercial text-to-image (T2I) models such as DALL-E 3. To the best of our knowledge, this marks 
    
[^2]: 在联邦学习中评估成员推断攻击和防御

    Evaluating Membership Inference Attacks and Defenses in Federated Learning

    [https://arxiv.org/abs/2402.06289](https://arxiv.org/abs/2402.06289)

    这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。

    

    成员推断攻击(MIAs)对于隐私保护的威胁在联邦学习中日益增长。半诚实的攻击者，例如服务器，可以根据观察到的模型信息确定一个特定样本是否属于目标客户端。本文对现有的MIAs和相应的防御策略进行了评估。我们对MIAs的评估揭示了两个重要发现。首先，结合多个通信轮次的模型信息(多时序)相比于利用单个时期的模型信息提高了MIAs的整体有效性。其次，在非目标客户端(Multi-spatial)中融入模型显著提高了MIAs的效果，特别是当客户端的数据是同质的时候。这凸显了在MIAs中考虑时序和空间模型信息的重要性。接下来，我们通过隐私-效用权衡评估了两种类型的防御机制对MIAs的有效性。

    Membership Inference Attacks (MIAs) pose a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch. Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MI
    
[^3]: 基于深度神经网络的自适应巡航控制在上下文感知攻击下的安全性实验分析

    Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks. (arXiv:2307.08939v1 [cs.CR])

    [http://arxiv.org/abs/2307.08939](http://arxiv.org/abs/2307.08939)

    这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。

    

    自适应巡航控制（ACC）是一种广泛应用的驾驶员辅助功能，用于保持期望速度和与前方车辆的安全距离。本文评估基于深度神经网络（DNN）的ACC系统在隐蔽感知攻击下的安全性，该攻击会对摄像机数据进行有针对性的扰动，以导致前方碰撞事故。我们提出了一种基于知识和数据驱动的方法，设计了一种上下文感知策略，用于选择触发攻击最关键的时间点，并采用了一种新颖的基于优化的方法，在运行时生成适应性图像扰动。我们使用实际驾驶数据集和逼真的仿真平台评估了所提出攻击的有效性，该仿真平台使用了来自生产ACC系统的控制软件和物理世界驾驶模拟器，并考虑了驾驶员的干预以及自动紧急制动（AEB）和前向碰撞警示（FCW）等安全功能。

    Adaptive Cruise Control (ACC) is a widely used driver assistance feature for maintaining desired speed and safe distance to the leading vehicles. This paper evaluates the security of the deep neural network (DNN) based ACC systems under stealthy perception attacks that strategically inject perturbations into camera data to cause forward collisions. We present a combined knowledge-and-data-driven approach to design a context-aware strategy for the selection of the most critical times for triggering the attacks and a novel optimization-based method for the adaptive generation of image perturbations at run-time. We evaluate the effectiveness of the proposed attack using an actual driving dataset and a realistic simulation platform with the control software from a production ACC system and a physical-world driving simulator while considering interventions by the driver and safety features such as Automatic Emergency Braking (AEB) and Forward Collision Warning (FCW). Experimental results sh
    

