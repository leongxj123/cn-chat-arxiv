# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QNCD: Quantization Noise Correction for Diffusion Models](https://arxiv.org/abs/2403.19140) | 研究提出了一个统一的量化噪声校正方案（QNCD），旨在减小扩散模型中的量化噪声，解决了后训练量化对采样加速的影响问题。 |
| [^2] | [Tracking-Assisted Object Detection with Event Cameras](https://arxiv.org/abs/2403.18330) | 本文利用跟踪策略和自动标记算法，针对事件相机中的难以辨识的对象，揭示其特征信息。 |
| [^3] | [Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning.](http://arxiv.org/abs/2401.06344) | 本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。 |
| [^4] | [Continual Learning: Forget-free Winning Subnetworks for Video Representations.](http://arxiv.org/abs/2312.11973) | 本研究基于"彩票票据假设"，提出了一种连续学习方法，通过利用稀疏子网络和FSO进行任务增量学习、少样本类增量学习和视频增量学习，实现高效学习和有效的权重重用。 |
| [^5] | [NN-Copula-CD: A Copula-Guided Interpretable Neural Network for Change Detection in Heterogeneous Remote Sensing Images.](http://arxiv.org/abs/2303.17448) | 该论文提出了一种可解释的神经网络方法，结合Copula理论来解决异构遥感图像中的变化检测问题。 |

# 详细

[^1]: QNCD：扩散模型的量化噪声校正

    QNCD: Quantization Noise Correction for Diffusion Models

    [https://arxiv.org/abs/2403.19140](https://arxiv.org/abs/2403.19140)

    研究提出了一个统一的量化噪声校正方案（QNCD），旨在减小扩散模型中的量化噪声，解决了后训练量化对采样加速的影响问题。

    

    扩散模型在图像合成领域引起了革命，建立了质量和创造力的新基准。然而，它们在迭代去噪过程中需要的密集计算阻碍了它们的广泛采用。后训练量化（PTQ）提供了一种解决方案，可以加速采样，尽管以低比特设置极大降低了样本质量。针对这一问题，我们的研究引入了一个统一的量化噪声校正方案（QNCD），旨在减小整个采样过程中的量化噪声。我们确定了两个主要的量化挑战：内部和外部量化噪声。内部量化噪声主要由于嵌入在resblock模块中而加剧，扩展了激活量化范围，在每个单独的去噪步骤中增加了干扰。此外，外部量化噪声源自整个去噪过程中的累积量化偏差，改变了数据分布。

    arXiv:2403.19140v1 Announce Type: cross  Abstract: Diffusion models have revolutionized image synthesis, setting new benchmarks in quality and creativity. However, their widespread adoption is hindered by the intensive computation required during the iterative denoising process. Post-training quantization (PTQ) presents a solution to accelerate sampling, aibeit at the expense of sample quality, extremely in low-bit settings. Addressing this, our study introduces a unified Quantization Noise Correction Scheme (QNCD), aimed at minishing quantization noise throughout the sampling process. We identify two primary quantization challenges: intra and inter quantization noise. Intra quantization noise, mainly exacerbated by embeddings in the resblock module, extends activation quantization ranges, increasing disturbances in each single denosing step. Besides, inter quantization noise stems from cumulative quantization deviations across the entire denoising process, altering data distributions 
    
[^2]: 使用跟踪辅助的事件相机目标检测

    Tracking-Assisted Object Detection with Event Cameras

    [https://arxiv.org/abs/2403.18330](https://arxiv.org/abs/2403.18330)

    本文利用跟踪策略和自动标记算法，针对事件相机中的难以辨识的对象，揭示其特征信息。

    

    最近，由于事件相机具有高动态范围和无动态模糊等特殊属性，事件驱动目标检测在计算机视觉领域引起了关注。然而，由于特征的不同步性和稀疏性导致了由于相机与之没有相对运动而导致的看不见的对象，这对任务构成了重大挑战。本文将这些看不见的对象视为伪遮挡对象，并旨在揭示其特征。首先，我们引入了对象的可见性属性，并提出了一个自动标记算法，用于在现有的事件相机数据集上附加额外的可见性标签。其次，我们利用跟踪策略来

    arXiv:2403.18330v1 Announce Type: cross  Abstract: Event-based object detection has recently garnered attention in the computer vision community due to the exceptional properties of event cameras, such as high dynamic range and no motion blur. However, feature asynchronism and sparsity cause invisible objects due to no relative motion to the camera, posing a significant challenge in the task. Prior works have studied various memory mechanisms to preserve as many features as possible at the current time, guided by temporal clues. While these implicit-learned memories retain some short-term information, they still struggle to preserve long-term features effectively. In this paper, we consider those invisible objects as pseudo-occluded objects and aim to reveal their features. Firstly, we introduce visibility attribute of objects and contribute an auto-labeling algorithm to append additional visibility labels on an existing event camera dataset. Secondly, we exploit tracking strategies fo
    
[^3]: 超级-STTN：社交群体感知的时空转换网络用于人体轨迹预测与超图推理

    Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning. (arXiv:2401.06344v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2401.06344](http://arxiv.org/abs/2401.06344)

    本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    

    在各种现实世界的应用中，包括服务机器人和自动驾驶汽车，预测拥挤的意图和轨迹是至关重要的。理解环境动态是具有挑战性的，不仅因为对建模成对的空间和时间相互作用的复杂性，还因为群体间相互作用的多样性。为了解码拥挤场景中全面的成对和群体间相互作用，我们引入了Hyper-STTN，这是一种基于超图的时空转换网络，用于人群轨迹预测。在Hyper-STTN中，通过一组多尺度超图构建了拥挤的群体间相关性，这些超图具有不同的群体大小，通过基于随机游走概率的超图谱卷积进行捕捉。此外，还采用了空间-时间转换器来捕捉行人在空间-时间维度上的对照相互作用。然后，这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    Predicting crowded intents and trajectories is crucial in varouls real-world applications, including service robots and autonomous vehicles. Understanding environmental dynamics is challenging, not only due to the complexities of modeling pair-wise spatial and temporal interactions but also the diverse influence of group-wise interactions. To decode the comprehensive pair-wise and group-wise interactions in crowded scenarios, we introduce Hyper-STTN, a Hypergraph-based Spatial-Temporal Transformer Network for crowd trajectory prediction. In Hyper-STTN, crowded group-wise correlations are constructed using a set of multi-scale hypergraphs with varying group sizes, captured through random-walk robability-based hypergraph spectral convolution. Additionally, a spatial-temporal transformer is adapted to capture pedestrians' pair-wise latent interactions in spatial-temporal dimensions. These heterogeneous group-wise and pair-wise are then fused and aligned though a multimodal transformer net
    
[^4]: 连续学习: 面向视频表示的免遗忘优胜子网络

    Continual Learning: Forget-free Winning Subnetworks for Video Representations. (arXiv:2312.11973v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.11973](http://arxiv.org/abs/2312.11973)

    本研究基于"彩票票据假设"，提出了一种连续学习方法，通过利用稀疏子网络和FSO进行任务增量学习、少样本类增量学习和视频增量学习，实现高效学习和有效的权重重用。

    

    受到"彩票票据假设"（LTH）的启发，该假设强调在较大的密集网络中存在高效子网络，研究了在适当的稀疏条件下表现优秀的优胜子网络（WSN）在各种连续学习任务中的应用。它利用来自密集网络的预先存在的权重，在任务增量学习（TIL）场景中实现高效学习。在少样本类增量学习（FSCIL）中，设计了一种称为软子网络（SoftNet）的WSN变体，以防止数据样本稀缺时的过拟合。此外，考虑了WSN权重的稀疏重用，用于视频增量学习（VIL）。考虑了在WSN中使用傅立叶子神经运算器（FSO），它能够对视频进行紧凑编码，并在不同带宽下识别可重用的子网络。我们将FSO集成到不同的连续学习架构中，包括VIL、TIL和FSCIL。

    Inspired by the Lottery Ticket Hypothesis (LTH), which highlights the existence of efficient subnetworks within larger, dense networks, a high-performing Winning Subnetwork (WSN) in terms of task performance under appropriate sparsity conditions is considered for various continual learning tasks. It leverages pre-existing weights from dense networks to achieve efficient learning in Task Incremental Learning (TIL) scenarios. In Few-Shot Class Incremental Learning (FSCIL), a variation of WSN referred to as the Soft subnetwork (SoftNet) is designed to prevent overfitting when the data samples are scarce. Furthermore, the sparse reuse of WSN weights is considered for Video Incremental Learning (VIL). The use of Fourier Subneural Operator (FSO) within WSN is considered. It enables compact encoding of videos and identifies reusable subnetworks across varying bandwidths. We have integrated FSO into different architectural frameworks for continual learning, including VIL, TIL, and FSCIL. Our c
    
[^5]: NN-Copula-CD：一种基于Copula的可解释神经网络用于异构遥感图像变化检测

    NN-Copula-CD: A Copula-Guided Interpretable Neural Network for Change Detection in Heterogeneous Remote Sensing Images. (arXiv:2303.17448v1 [cs.CV])

    [http://arxiv.org/abs/2303.17448](http://arxiv.org/abs/2303.17448)

    该论文提出了一种可解释的神经网络方法，结合Copula理论来解决异构遥感图像中的变化检测问题。

    

    异构遥感图像中的变化检测是一个实际而具有挑战性的问题。过去十年来，深度神经网络(DNN)的发展让异构变化检测问题受益匪浅。然而，数据驱动的DNN始终像黑匣子一样，缺乏可解释性，这限制了DNN在大多数实际变化检测应用中的可靠性和可控性。为了解决这些问题，我们提出了一种基于Copula的可解释神经网络异构变化检测方法(NN-Copula-CD)。在NN-Copula-CD中，Copula的数学特征被设计为损失函数，用于监督一个简单的全连接神经网络学习变量之间的相关性。

    Change detection (CD) in heterogeneous remote sensing images is a practical and challenging issue for real-life emergencies. In the past decade, the heterogeneous CD problem has significantly benefited from the development of deep neural networks (DNN). However, the data-driven DNNs always perform like a black box where the lack of interpretability limits the trustworthiness and controllability of DNNs in most practical CD applications. As a strong knowledge-driven tool to measure correlation between random variables, Copula theory has been introduced into CD, yet it suffers from non-robust CD performance without manual prior selection for Copula functions. To address the above issues, we propose a knowledge-data-driven heterogeneous CD method (NN-Copula-CD) based on the Copula-guided interpretable neural network. In our NN-Copula-CD, the mathematical characteristics of Copula are designed as the losses to supervise a simple fully connected neural network to learn the correlation betwe
    

