# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness.](http://arxiv.org/abs/2312.04960) | MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。 |
| [^2] | [LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving.](http://arxiv.org/abs/2310.03026) | 本文研究将大型语言模型（LLMs）作为复杂自动驾驶场景的决策组件，通过认知路径和算法来实现全面推理和可执行驾驶指令的转化。实验证明，LLMs能够在单车任务和复杂驾驶行为中表现出优越性能，这是因为其具有常识推理能力。 |
| [^3] | [Awesome-META+: Meta-Learning Research and Learning Platform.](http://arxiv.org/abs/2304.12921) | Awesome-META+是一个元学习框架集成和学习平台，旨在提供完整可靠的元学习框架应用和面向初学者的学习材料，进而促进元学习的发展并将其从小众领域转化为主流的研究方向。 |
| [^4] | [Privacy-Preserving CNN Training with Transfer Learning.](http://arxiv.org/abs/2304.03807) | 本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。 |

# 详细

[^1]: MIMIR: 基于互信息的对抗鲁棒性的遮罩图像建模

    MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness. (arXiv:2312.04960v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.04960](http://arxiv.org/abs/2312.04960)

    MIMIR提出了一种新颖的防御方法，通过在预训练中利用遮罩图像建模，构建了一个不同的对抗性训练方法。该方法旨在增强Vision Transformers（ViTs）对抗攻击的鲁棒性。

    

    视觉变压器（ViTs）相对于卷积神经网络（CNNs）在各种任务上实现了卓越的性能，但ViTs也容易受到对抗性攻击。对抗性训练是建立强大的CNN模型的最成功方法之一。因此，最近的研究探索了基于ViTs和CNNs之间的差异的对抗性训练的新方法，如更好的训练策略，防止注意力集中在单个块上，或丢弃低注意力的嵌入。然而，这些方法仍然遵循传统监督对抗训练的设计，限制了对ViTs的对抗训练的潜力。本文提出了一种新颖的防御方法MIMIR，旨在通过利用预训练中的遮罩图像建模构建不同的对抗性训练方法。我们创建了一个自编码器，它接受对抗性例子作为输入，但将干净的例子作为建模目标。然后，我们创建了一个互信息（MI）

    Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI
    
[^2]: LanguageMPC：基于大型语言模型的自动驾驶决策者

    LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving. (arXiv:2310.03026v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2310.03026](http://arxiv.org/abs/2310.03026)

    本文研究将大型语言模型（LLMs）作为复杂自动驾驶场景的决策组件，通过认知路径和算法来实现全面推理和可执行驾驶指令的转化。实验证明，LLMs能够在单车任务和复杂驾驶行为中表现出优越性能，这是因为其具有常识推理能力。

    

    现有基于学习的自动驾驶系统在理解高级信息、推广罕见事件和提供可解释性方面面临挑战。为解决这些问题，本研究将大型语言模型（LLMs）作为复杂自动驾驶场景的决策组件，需要人类常识理解。我们设计了认知路径，使LLMs能够进行全面推理，并开发了将LLM决策转化为可执行驾驶指令的算法。通过这种方式，LLM决策通过引导参数矩阵适应与低级控制器无缝集成。大量实验表明，我们提出的方法不仅在单车任务中始终超越基线方法，而且还能处理复杂的驾驶行为，甚至多车协调，这要归功于LLMs的常识推理能力。本文介绍了将LLMs作为有效决策者的初步步骤。

    Existing learning-based autonomous driving (AD) systems face challenges in comprehending high-level information, generalizing to rare events, and providing interpretability. To address these problems, this work employs Large Language Models (LLMs) as a decision-making component for complex AD scenarios that require human commonsense understanding. We devise cognitive pathways to enable comprehensive reasoning with LLMs, and develop algorithms for translating LLM decisions into actionable driving commands. Through this approach, LLM decisions are seamlessly integrated with low-level controllers by guided parameter matrix adaptation. Extensive experiments demonstrate that our proposed method not only consistently surpasses baseline approaches in single-vehicle tasks, but also helps handle complex driving behaviors even multi-vehicle coordination, thanks to the commonsense reasoning capabilities of LLMs. This paper presents an initial step toward leveraging LLMs as effective decision-make
    
[^3]: Awesome-META+: 元学习研究与学习平台

    Awesome-META+: Meta-Learning Research and Learning Platform. (arXiv:2304.12921v1 [cs.LG])

    [http://arxiv.org/abs/2304.12921](http://arxiv.org/abs/2304.12921)

    Awesome-META+是一个元学习框架集成和学习平台，旨在提供完整可靠的元学习框架应用和面向初学者的学习材料，进而促进元学习的发展并将其从小众领域转化为主流的研究方向。

    

    人工智能已经在经济、产业、教育等各个领域产生了深远的影响，但还存在诸多限制。元学习，也称为“学习如何学习”，为通用人工智能提供了突破目前瓶颈的机会。然而，元学习起步较晚，相比CV、NLP等领域，项目数量较少。每次部署都需要大量的经验去配置环境、调试代码甚至重写，而且框架之间相对孤立。此外，目前针对元学习的专门平台和面向初学者的学习材料相对较少，门槛相对较高。基于此，Awesome-META+提出了一个元学习框架集成和学习平台，旨在解决上述问题并提供完整可靠的元学习框架应用和学习平台。该项目旨在促进元学习的发展，并将其从一个小众领域转化为一个主流的研究方向。

    Artificial intelligence technology has already had a profound impact in various fields such as economy, industry, and education, but still limited. Meta-learning, also known as "learning to learn", provides an opportunity for general artificial intelligence, which can break through the current AI bottleneck. However, meta learning started late and there are fewer projects compare with CV, NLP etc. Each deployment requires a lot of experience to configure the environment, debug code or even rewrite, and the frameworks are isolated. Moreover, there are currently few platforms that focus exclusively on meta-learning, or provide learning materials for novices, for which the threshold is relatively high. Based on this, Awesome-META+, a meta-learning framework integration and learning platform is proposed to solve the above problems and provide a complete and reliable meta-learning framework application and learning platform. The project aims to promote the development of meta-learning and t
    
[^4]: 使用迁移学习实现隐私保护的CNN训练

    Privacy-Preserving CNN Training with Transfer Learning. (arXiv:2304.03807v1 [cs.CR])

    [http://arxiv.org/abs/2304.03807](http://arxiv.org/abs/2304.03807)

    本文提出了一种使用迁移学习实现同态加密技术下隐私保护的CNN训练的方案，通过转换思想和更快的梯度变体，取得了最先进的性能。

    

    隐私保护的神经网络推理已经得到很好的研究，同时保持同态CNN训练仍然是一项挑战性的任务。在本文中，我们提出了一种实用的解决方案来实现基于同态加密技术的隐私保护CNN训练。据我们所知，这是第一次成功突破这个难题，以前没有任何工作达到这个目标。采用了几种技术：（1）通过迁移学习，可以将隐私保护的CNN训练简化为同态神经网络训练，甚至是多类逻辑回归（MLR）训练；（2）通过更快的梯度变体$\texttt{Quadratic Gradient}$，应用于MLR的增强梯度方法，在收敛速度方面具有最先进的性能；（3）我们采用数学中的变换思想，将加密域中的近似Softmax函数转换成已经研究过的逼近方法，从而得到更好的结果。

    Privacy-preserving nerual network inference has been well studied while homomorphic CNN training still remains an open challenging task. In this paper, we present a practical solution to implement privacy-preserving CNN training based on mere Homomorphic Encryption (HE) technique. To our best knowledge, this is the first attempt successfully to crack this nut and no work ever before has achieved this goal. Several techniques combine to make it done: (1) with transfer learning, privacy-preserving CNN training can be reduced to homomorphic neural network training, or even multiclass logistic regression (MLR) training; (2) via a faster gradient variant called $\texttt{Quadratic Gradient}$, an enhanced gradient method for MLR with a state-of-the-art performance in converge speed is applied in this work to achieve high performance; (3) we employ the thought of transformation in mathematics to transform approximating Softmax function in encryption domain to the well-studied approximation of 
    

