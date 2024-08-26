# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^2] | [Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging](https://arxiv.org/abs/2403.11865) | 本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。 |
| [^3] | [Self-Supervised Multiple Instance Learning for Acute Myeloid Leukemia Classification](https://arxiv.org/abs/2403.05379) | 自本研究发现自监督预训练编码器在多实例学习中实现了可比较的性能，展示了自监督学习在急性髓细胞白血病分类中的潜力，这为一种经济高效且节约数据的解决方案。 |
| [^4] | [Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization](https://arxiv.org/abs/2401.16352) | 提出了一种新的对净化的对抗训练（AToP）流程，通过随机转换的扰动破坏和通过对抗损失微调净化器模型，同时提升了鲁棒性和泛化性能。 |
| [^5] | [EdgeOL: Efficient in-situ Online Learning on Edge Devices.](http://arxiv.org/abs/2401.16694) | 本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。 |
| [^6] | [A Geometric Perspective on Diffusion Models.](http://arxiv.org/abs/2305.19947) | 本文研究了扩散模型的几何结构，发现通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接了数据分布和噪声分布，建立了基于ODE的最优采样和经典的均值漂移算法之间的理论关系。 |
| [^7] | [Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery.](http://arxiv.org/abs/2303.15975) | 本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。 |

# 详细

[^1]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^2]: 利用热成像探索多模态神经场景表示并应用

    Exploring Multi-modal Neural Scene Representations With Applications on Thermal Imaging

    [https://arxiv.org/abs/2403.11865](https://arxiv.org/abs/2403.11865)

    本文对神经场景表示在多模态学习中的应用进行了全面评估，并提出了四种不同的策略以将第二模态（非RGB）纳入NeRFs中，通过选择热成像作为第二模态来挑战神经场景表示的整合。

    

    神经辐射场（NeRFs）在一组RGB图像上训练时迅速发展为新的事实标准，用于新视角合成任务。本文在多模态学习的背景下对神经场景表示（如NeRFs）进行了全面评估。具体而言，我们提出了四种不同的策略，用于如何将第二模态（非RGB）纳入NeRFs中：（1）独立地从头训练每种模态；（2）在RGB上进行预训练，然后在第二模态上进行微调；（3）添加第二分支；（4）添加一个单独的组件来预测（颜色）额外模态的值。我们选择了热成像作为第二模态，因为从辐射度来看，它与RGB有很大差异，这使得将其整合到神经场景表示中具有挑战性。

    arXiv:2403.11865v1 Announce Type: cross  Abstract: Neural Radiance Fields (NeRFs) quickly evolved as the new de-facto standard for the task of novel view synthesis when trained on a set of RGB images. In this paper, we conduct a comprehensive evaluation of neural scene representations, such as NeRFs, in the context of multi-modal learning. Specifically, we present four different strategies of how to incorporate a second modality, other than RGB, into NeRFs: (1) training from scratch independently on both modalities; (2) pre-training on RGB and fine-tuning on the second modality; (3) adding a second branch; and (4) adding a separate component to predict (color) values of the additional modality. We chose thermal imaging as second modality since it strongly differs from RGB in terms of radiosity, making it challenging to integrate into neural scene representations. For the evaluation of the proposed strategies, we captured a new publicly available multi-view dataset, ThermalMix, consisti
    
[^3]: 自监督多实例学习用于急性髓细胞白血病分类

    Self-Supervised Multiple Instance Learning for Acute Myeloid Leukemia Classification

    [https://arxiv.org/abs/2403.05379](https://arxiv.org/abs/2403.05379)

    自本研究发现自监督预训练编码器在多实例学习中实现了可比较的性能，展示了自监督学习在急性髓细胞白血病分类中的潜力，这为一种经济高效且节约数据的解决方案。

    

    自动疾病诊断使用医学图像分析依赖深度学习，通常需要大量标记数据集进行监督模型训练。急性髓细胞白血病（AML）等疾病由于在单个细胞水平上稀缺且昂贵的标注而面临挑战。多实例学习（MIL）解决了弱标记场景，但通常需要用标记数据训练的强大编码器。在本研究中，我们探索了自监督学习（SSL）作为基于MIL的AML亚型分类的预训练方法，从血涂片中去除了编码器训练期间的标记数据需求。我们研究了三种最先进的SSL方法SimCLR、SwAV和DINO，并将它们的性能与监督预训练进行了比较。我们的研究结果表明，SSL预训练编码器实现了可比较的性能，展示了SSL在MIL中的潜力。这一突破提供了一种经济高效且节约数据的解决方案，

    arXiv:2403.05379v1 Announce Type: cross  Abstract: Automated disease diagnosis using medical image analysis relies on deep learning, often requiring large labeled datasets for supervised model training. Diseases like Acute Myeloid Leukemia (AML) pose challenges due to scarce and costly annotations on a single-cell level. Multiple Instance Learning (MIL) addresses weakly labeled scenarios but necessitates powerful encoders typically trained with labeled data. In this study, we explore Self-Supervised Learning (SSL) as a pre-training approach for MIL-based AML subtype classification from blood smears, removing the need for labeled data during encoder training. We investigate the three state-of-the-art SSL methods SimCLR, SwAV, and DINO, and compare their performance against supervised pre-training. Our findings show that SSL-pretrained encoders achieve comparable performance, showcasing the potential of SSL in MIL. This breakthrough offers a cost-effective and data-efficient solution, pr
    
[^4]: 对净化的对抗训练（AToP）：提升鲁棒性和泛化性能

    Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization

    [https://arxiv.org/abs/2401.16352](https://arxiv.org/abs/2401.16352)

    提出了一种新的对净化的对抗训练（AToP）流程，通过随机转换的扰动破坏和通过对抗损失微调净化器模型，同时提升了鲁棒性和泛化性能。

    

    深度神经网络被认为易受设计精良的对抗攻击影响。基于对抗训练（AT）的最成功防御技术可以实现特定攻击下的最佳鲁棒性，但无法很好地泛化到未知攻击。基于对抗净化（AP）的另一有效防御技术可以增强泛化性能，但无法实现最佳鲁棒性。与此同时，这两种方法都存在一个共同的局限性，即标准准确性降级。为了缓解这些问题，我们提出了一种新的流程，称为对净化的对抗训练（AToP），包括两个组件：通过随机转换（RT）破坏扰动，以避免对已知攻击的过度学习，从而实现对未知攻击的鲁棒性泛化；以及通过对抗损失对净化器模型进行微调（FT），以提高鲁棒性。为了评估我们的方法，我们在一种...

    arXiv:2401.16352v2 Announce Type: replace-cross  Abstract: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an e
    
[^5]: EdgeOL: 边缘设备上高效的原位在线学习

    EdgeOL: Efficient in-situ Online Learning on Edge Devices. (arXiv:2401.16694v1 [cs.LG])

    [http://arxiv.org/abs/2401.16694](http://arxiv.org/abs/2401.16694)

    本文提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率，在边缘设备上实现了显著的性能提升。

    

    新兴应用，如机器人辅助养老和物体识别，通常采用深度学习神经网络模型，并且自然需要：i) 处理实时推理请求和ii) 适应可能的部署场景变化。在线模型微调被广泛采用以满足这些需求。然而，微调会导致显著的能量消耗，使其难以部署在边缘设备上。在本文中，我们提出了EdgeOL，一种边缘在线学习框架，通过内部和外部调优来优化推理准确性、微调执行时间和能量效率。实验结果显示，EdgeOL平均减少了82%的微调执行时间，74%的能量消耗，并提高了平均推理准确率1.70%，相对于即时在线学习策略。

    Emerging applications, such as robot-assisted eldercare and object recognition, generally employ deep learning neural networks (DNNs) models and naturally require: i) handling streaming-in inference requests and ii) adapting to possible deployment scenario changes. Online model fine-tuning is widely adopted to satisfy these needs. However, fine-tuning involves significant energy consumption, making it challenging to deploy on edge devices. In this paper, we propose EdgeOL, an edge online learning framework that optimizes inference accuracy, fine-tuning execution time, and energy efficiency through both inter-tuning and intra-tuning optimizations. Experimental results show that, on average, EdgeOL reduces overall fine-tuning execution time by 82%, energy consumption by 74%, and improves average inference accuracy by 1.70% over the immediate online learning strategy.
    
[^6]: 扩散模型的几何视角

    A Geometric Perspective on Diffusion Models. (arXiv:2305.19947v1 [cs.CV])

    [http://arxiv.org/abs/2305.19947](http://arxiv.org/abs/2305.19947)

    本文研究了扩散模型的几何结构，发现通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接了数据分布和噪声分布，建立了基于ODE的最优采样和经典的均值漂移算法之间的理论关系。

    

    近年来，针对扩散模型的高效训练和快速采样方法取得了显著进展。最近的一个重要进展是使用随机微分方程（SDE）来描述数据扰动和生成建模，以实现统一的数学框架。本文揭示了扩散模型的几个有趣的几何结构，并为其采样动力学提供了简单而强大的解释。通过仔细检查一种流行的方差爆炸SDE及其保持边际的普通微分方程（ODE）用于采样，我们发现数据分布和噪声分布通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接，即使在视觉质量方面也收敛更快。我们还建立起基于ODE的最优采样和经典的均值漂移（寻找模式）算法之间的理论关系。

    Recent years have witnessed significant progress in developing efficient training and fast sampling approaches for diffusion models. A recent remarkable advancement is the use of stochastic differential equations (SDEs) to describe data perturbation and generative modeling in a unified mathematical framework. In this paper, we reveal several intriguing geometric structures of diffusion models and contribute a simple yet powerful interpretation to their sampling dynamics. Through carefully inspecting a popular variance-exploding SDE and its marginal-preserving ordinary differential equation (ODE) for sampling, we discover that the data distribution and the noise distribution are smoothly connected with an explicit, quasi-linear sampling trajectory, and another implicit denoising trajectory, which even converges faster in terms of visual quality. We also establish a theoretical relationship between the optimal ODE-based sampling and the classic mean-shift (mode-seeking) algorithm, with w
    
[^7]: 大规模预训练模型在增量式新类别发现中具有出乎意料的强大表现。

    Large-scale Pre-trained Models are Surprisingly Strong in Incremental Novel Class Discovery. (arXiv:2303.15975v1 [cs.CV])

    [http://arxiv.org/abs/2303.15975](http://arxiv.org/abs/2303.15975)

    本论文提出了一种更加挑战性和实用性的学习方法MSc-iNCD，通过在连续而无人监督的学习中利用大规模预训练模型的丰富先验知识，该方法在增量式新类别发现中表现出出乎意料的强大实力。

    

    在生命长学习者中，从未标记的数据中连续地发现新概念是一个重要的期望。在文献中，这类问题在非常受限的情况下得到了部分解决，其中要么为发现新概念提供有标号的数据（例如 NCD），要么学习在有限数量的增量步骤中发生（例如类 iNCD）。在这项工作中，我们挑战现状，提出了一种更具挑战性和实用性的学习范式，称为 MSc-iNCD，其中学习连续而无人监督，并利用大规模预训练模型的丰富先验知识。为此，我们提出了简单的基线，不仅在较长的学习情境下具有弹性，而且与复杂的最先进方法相比，表现出出乎意料的强大实力。我们在多个基准测试中进行了广泛的实证评估，并展示了我们提出的基线的有效性，大大提升了基准要求。

    Discovering novel concepts from unlabelled data and in a continuous manner is an important desideratum of lifelong learners. In the literature such problems have been partially addressed under very restricted settings, where either access to labelled data is provided for discovering novel concepts (e.g., NCD) or learning occurs for a limited number of incremental steps (e.g., class-iNCD). In this work we challenge the status quo and propose a more challenging and practical learning paradigm called MSc-iNCD, where learning occurs continuously and unsupervisedly, while exploiting the rich priors from large-scale pre-trained models. To this end, we propose simple baselines that are not only resilient under longer learning scenarios, but are surprisingly strong when compared with sophisticated state-of-the-art methods. We conduct extensive empirical evaluation on a multitude of benchmarks and show the effectiveness of our proposed baselines, which significantly raises the bar.
    

