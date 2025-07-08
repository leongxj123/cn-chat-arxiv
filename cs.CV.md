# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377) | 提出了一种名为扰动注意力引导（PAG）的新型抽样引导技术，通过在扩散 U-Net 中替换自注意力映射来生成结构降级的中间样本，从而在无条件和有条件设置下改善扩散样本质量。 |
| [^2] | [VGMShield: Mitigating Misuse of Video Generative Models](https://arxiv.org/abs/2402.13126) | VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。 |
| [^3] | [PIP-Net: Pedestrian Intention Prediction in the Wild](https://arxiv.org/abs/2402.12810) | PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。 |
| [^4] | [Efficient generative adversarial networks using linear additive-attention Transformers.](http://arxiv.org/abs/2401.09596) | 这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。 |
| [^5] | [AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection.](http://arxiv.org/abs/2310.13103) | 本文提出了AVTENet框架，该框架是一个基于音频-视觉Transformer的多专家集成网络，用于在视频深度伪造检测中考虑声学和视觉操作。 |
| [^6] | [Uncertainty in Real-Time Semantic Segmentation on Embedded Systems.](http://arxiv.org/abs/2301.01201) | 本文提出了一种结合贝叶斯回归和动量传播的预测方法，能够实时在嵌入式硬件上产生有意义的不确定性，从而使语义分割模型在自动驾驶和人机交互等领域的实时应用成为可能。 |

# 详细

[^1]: 具有扰动注意力引导的自矫正扩散抽样

    Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance

    [https://arxiv.org/abs/2403.17377](https://arxiv.org/abs/2403.17377)

    提出了一种名为扰动注意力引导（PAG）的新型抽样引导技术，通过在扩散 U-Net 中替换自注意力映射来生成结构降级的中间样本，从而在无条件和有条件设置下改善扩散样本质量。

    

    近期研究表明，扩散模型能够生成高质量样本，但其质量很大程度上依赖于抽样引导技术，比如分类器引导（CG）和无分类器引导（CFG）。这些技术通常在无条件生成或各种下游任务如图像恢复中无法应用。本文提出了一种新颖的抽样引导技术，称为扰动注意力引导（PAG），它改进了扩散样本的质量，不管是在无条件还是有条件的设置中，都能实现这一目标，而不需要额外训练或整合外部模块。PAG 旨在通过整个去噪过程逐步增强样本的结构。它涉及通过用恒等矩阵替换扩散 U-Net 中选择的自注意力映射生成结构降级的中间样本，考虑自注意力机制。

    arXiv:2403.17377v1 Announce Type: cross  Abstract: Recent studies have demonstrated that diffusion models are capable of generating high-quality samples, but their quality heavily depends on sampling guidance techniques, such as classifier guidance (CG) and classifier-free guidance (CFG). These techniques are often not applicable in unconditional generation or in various downstream tasks such as image restoration. In this paper, we propose a novel sampling guidance, called Perturbed-Attention Guidance (PAG), which improves diffusion sample quality across both unconditional and conditional settings, achieving this without requiring additional training or the integration of external modules. PAG is designed to progressively enhance the structure of samples throughout the denoising process. It involves generating intermediate samples with degraded structure by substituting selected self-attention maps in diffusion U-Net with an identity matrix, by considering the self-attention mechanisms
    
[^2]: VGMShield：缓解视频生成模型的误用

    VGMShield: Mitigating Misuse of Video Generative Models

    [https://arxiv.org/abs/2402.13126](https://arxiv.org/abs/2402.13126)

    VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。

    

    随着视频生成技术的快速发展，人们可以方便地利用视频生成模型创建符合其特定需求的视频。然而，人们也越来越担心这些技术被用于创作和传播虚假信息。在这项工作中，我们介绍了VGMShield：一套包含三项直接但开创性的措施，用于防范虚假视频生成过程中可能出现的问题。我们首先从“虚假视频检测”开始，尝试理解生成的视频中是否存在独特性，以及我们是否能够区分它们与真实视频的不同；然后，我们探讨“溯源”问题，即将一段虚假视频追溯回生成它的模型。为此，我们提出利用预训练的关注“时空动态”的模型作为骨干，以识别视频中的不一致性。通过对七个最先进的开源模型进行实验，我们证明了...

    arXiv:2402.13126v1 Announce Type: cross  Abstract: With the rapid advancement in video generation, people can conveniently utilize video generation models to create videos tailored to their specific desires. Nevertheless, there are also growing concerns about their potential misuse in creating and disseminating false information.   In this work, we introduce VGMShield: a set of three straightforward but pioneering mitigations through the lifecycle of fake video generation. We start from \textit{fake video detection} trying to understand whether there is uniqueness in generated videos and whether we can differentiate them from real videos; then, we investigate the \textit{tracing} problem, which maps a fake video back to a model that generates it. Towards these, we propose to leverage pre-trained models that focus on {\it spatial-temporal dynamics} as the backbone to identify inconsistencies in videos. Through experiments on seven state-of-the-art open-source models, we demonstrate that
    
[^3]: PIP-Net：城市中行人意图预测

    PIP-Net: Pedestrian Intention Prediction in the Wild

    [https://arxiv.org/abs/2402.12810](https://arxiv.org/abs/2402.12810)

    PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。

    

    精准的自动驾驶车辆（AVs）对行人意图的预测是当前该领域的一项研究挑战。在本文中，我们介绍了PIP-Net，这是一个新颖的框架，旨在预测AVs在现实世界城市场景中的行人过马路意图。我们提供了两种针对不同摄像头安装和设置设计的PIP-Net变种。利用来自行驶场景的动力学数据和空间特征，所提出的模型采用循环和时间注意力机制的解决方案，性能优于现有技术。为了增强道路用户的视觉表示及其与自车的相关性，我们引入了一个分类深度特征图，结合局部运动流特征，为场景动态提供丰富的洞察。此外，我们探讨了将摄像头的视野从一个扩展到围绕自车的三个摄像头的影响，以提升

    arXiv:2402.12810v1 Announce Type: cross  Abstract: Accurate pedestrian intention prediction (PIP) by Autonomous Vehicles (AVs) is one of the current research challenges in this field. In this article, we introduce PIP-Net, a novel framework designed to predict pedestrian crossing intentions by AVs in real-world urban scenarios. We offer two variants of PIP-Net designed for different camera mounts and setups. Leveraging both kinematic data and spatial features from the driving scene, the proposed model employs a recurrent and temporal attention-based solution, outperforming state-of-the-art performance. To enhance the visual representation of road users and their proximity to the ego vehicle, we introduce a categorical depth feature map, combined with a local motion flow feature, providing rich insights into the scene dynamics. Additionally, we explore the impact of expanding the camera's field of view, from one to three cameras surrounding the ego vehicle, leading to enhancement in the
    
[^4]: 使用线性加法注意力Transformer的高效生成对抗网络

    Efficient generative adversarial networks using linear additive-attention Transformers. (arXiv:2401.09596v1 [cs.CV])

    [http://arxiv.org/abs/2401.09596](http://arxiv.org/abs/2401.09596)

    这项工作提出了一种名为LadaGAN的高效生成对抗网络，它使用了一种名为Ladaformer的新型Transformer块，通过线性加法注意机制来降低计算复杂度并解决训练不稳定性问题。

    

    尽管像扩散模型（DMs）和生成对抗网络（GANs）等深度生成模型在图像生成方面的能力近年来得到了显著提高，但是它们的成功很大程度上归功于计算复杂的架构。这限制了它们在研究实验室和资源充足的公司中的采用和使用，同时也极大地增加了训练、微调和推理的碳足迹。在这项工作中，我们提出了LadaGAN，这是一个高效的生成对抗网络，它建立在一种名为Ladaformer的新型Transformer块上。该块的主要组成部分是一个线性加法注意机制，它每个头部计算一个注意向量，而不是二次的点积注意力。我们在生成器和判别器中都采用了Ladaformer，这降低了计算复杂度，并克服了Transformer GAN经常出现的训练不稳定性。LadaGAN一直表现优于现有的GANs。

    Although the capacity of deep generative models for image generation, such as Diffusion Models (DMs) and Generative Adversarial Networks (GANs), has dramatically improved in recent years, much of their success can be attributed to computationally expensive architectures. This has limited their adoption and use to research laboratories and companies with large resources, while significantly raising the carbon footprint for training, fine-tuning, and inference. In this work, we present LadaGAN, an efficient generative adversarial network that is built upon a novel Transformer block named Ladaformer. The main component of this block is a linear additive-attention mechanism that computes a single attention vector per head instead of the quadratic dot-product attention. We employ Ladaformer in both the generator and discriminator, which reduces the computational complexity and overcomes the training instabilities often associated with Transformer GANs. LadaGAN consistently outperforms exist
    
[^5]: AVTENet: 基于音频-视觉Transformer的多专家集成网络在视频深度伪造检测中的应用

    AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection. (arXiv:2310.13103v1 [cs.CV])

    [http://arxiv.org/abs/2310.13103](http://arxiv.org/abs/2310.13103)

    本文提出了AVTENet框架，该框架是一个基于音频-视觉Transformer的多专家集成网络，用于在视频深度伪造检测中考虑声学和视觉操作。

    

    在社交媒体平台上广泛分享的伪造内容是一个重大社会问题，要求加强监管并给研究社区带来新的挑战。近年来，超真实的深度伪造视频的普及引起了对音频和视觉伪造威胁的关注。大多数关于检测AI生成的伪造视频的先前工作只利用了视觉模态或音频模态。虽然文献中有一些方法利用音频和视觉模态来检测伪造视频，但它们尚未在涉及声学和视觉操作的多模态深度伪造视频数据集上进行全面评估。此外，这些现有方法大多基于CNN，并且检测准确率较低。受到Transformer在各个领域的最新成功启发，为了解决深度伪造技术带来的挑战，本文提出了一种考虑声学操作的音频-视觉Transformer集成网络（AVTENet）框架。

    Forged content shared widely on social media platforms is a major social problem that requires increased regulation and poses new challenges to the research community. The recent proliferation of hyper-realistic deepfake videos has drawn attention to the threat of audio and visual forgeries. Most previous work on detecting AI-generated fake videos only utilizes visual modality or audio modality. While there are some methods in the literature that exploit audio and visual modalities to detect forged videos, they have not been comprehensively evaluated on multi-modal datasets of deepfake videos involving acoustic and visual manipulations. Moreover, these existing methods are mostly based on CNN and suffer from low detection accuracy. Inspired by the recent success of Transformer in various fields, to address the challenges posed by deepfake technology, in this paper, we propose an Audio-Visual Transformer-based Ensemble Network (AVTENet) framework that considers both acoustic manipulatio
    
[^6]: 嵌入式系统实时语义分割中的不确定性

    Uncertainty in Real-Time Semantic Segmentation on Embedded Systems. (arXiv:2301.01201v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.01201](http://arxiv.org/abs/2301.01201)

    本文提出了一种结合贝叶斯回归和动量传播的预测方法，能够实时在嵌入式硬件上产生有意义的不确定性，从而使语义分割模型在自动驾驶和人机交互等领域的实时应用成为可能。

    

    语义分割模型在自动驾驶和人机交互等领域的应用需要实时预测能力。实时应用的挑战被资源受限的硬件所加剧。虽然这些平台上实时方法的开发得到了增加，但这些模型无法足够地考虑到存在的不确定性。本文通过将预先训练模型的深层特征提取与贝叶斯回归和动量传播相结合，提出了一种关注不确定性的预测方法。我们演示了该方法如何在嵌入式硬件上实时产生有意义的不确定性，同时保持预测性能。

    Application for semantic segmentation models in areas such as autonomous vehicles and human computer interaction require real-time predictive capabilities. The challenges of addressing real-time application is amplified by the need to operate on resource constrained hardware. Whilst development of real-time methods for these platforms has increased, these models are unable to sufficiently reason about uncertainty present. This paper addresses this by combining deep feature extraction from pre-trained models with Bayesian regression and moment propagation for uncertainty aware predictions. We demonstrate how the proposed method can yield meaningful uncertainty on embedded hardware in real-time whilst maintaining predictive performance.
    

