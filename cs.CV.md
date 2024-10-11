# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalized Consistency Trajectory Models for Image Manipulation](https://arxiv.org/abs/2403.12510) | 本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。 |
| [^2] | [Less is More: Data Value Estimation for Visual Instruction Tuning](https://arxiv.org/abs/2403.09559) | 视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。 |
| [^3] | [TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning](https://arxiv.org/abs/2402.19467) | TV-TREES是第一个多模态蕴涵树生成器，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树，实现了可解释联合模态推理，并在挑战性的TVQA数据集上展示了最先进的零-shot性能。 |
| [^4] | [SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT.](http://arxiv.org/abs/2307.01514) | 这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。 |
| [^5] | [OpenDriver: an open-road driver state detection dataset.](http://arxiv.org/abs/2304.04203) | OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。 |
| [^6] | [DualStreamFoveaNet: A Dual Stream Fusion Architecture with Anatomical Awareness for Robust Fovea Localization.](http://arxiv.org/abs/2302.06961) | DualStreamFoveaNet是一种具有解剖意识的双流融合架构，通过利用视网膜和血管分布进行多线索融合，实现对鲁棒的中央凹点定位。实验证明该架构在中央凹点定位方面达到了最先进的性能。 |

# 详细

[^1]: 图像操作的广义一致性轨迹模型

    Generalized Consistency Trajectory Models for Image Manipulation

    [https://arxiv.org/abs/2403.12510](https://arxiv.org/abs/2403.12510)

    本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。

    

    基于扩散的生成模型在无条件生成以及图像编辑和恢复等应用任务中表现出色。扩散模型的成功在于扩散的迭代性质：扩散将将噪声到数据的复杂映射过程分解为一系列简单的去噪任务。此外，通过在每个去噪步骤中注入引导项，我们能够对生成过程进行精细控制。然而，迭代过程也常常计算密集，通常需要进行数十次甚至数千次函数评估。虽然一致性轨迹模型（CTMs）可以在概率流ODE（PFODE）上任意时间点之间进行遍历，并且通过单次函数评估进行得分推导，但CTMs仅允许从高斯噪声转换为数据。因此，本文旨在通过提出广义CTMs（GCTMs）来发挥CTMs的全部潜力，实现在任何噪声分布和数据分布之间进行转换。

    arXiv:2403.12510v1 Announce Type: cross  Abstract: Diffusion-based generative models excel in unconditional generation, as well as on applied tasks such as image editing and restoration. The success of diffusion models lies in the iterative nature of diffusion: diffusion breaks down the complex process of mapping noise to data into a sequence of simple denoising tasks. Moreover, we are able to exert fine-grained control over the generation process by injecting guidance terms into each denoising step. However, the iterative process is also computationally intensive, often taking from tens up to thousands of function evaluations. Although consistency trajectory models (CTMs) enable traversal between any time points along the probability flow ODE (PFODE) and score inference with a single function evaluation, CTMs only allow translation from Gaussian noise to data. Thus, this work aims to unlock the full potential of CTMs by proposing generalized CTMs (GCTMs), which translate between arbit
    
[^2]: 数据价值评估对视觉指导调整的影响

    Less is More: Data Value Estimation for Visual Instruction Tuning

    [https://arxiv.org/abs/2403.09559](https://arxiv.org/abs/2403.09559)

    视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。

    

    视觉指导调整是构建多模式大语言模型（MLLMs）的关键，大大提高了大语言模型（LLMs）在视觉场景中的推理能力。然而，现有的MLLMs主要依赖于多个高度多样化的视觉指导数据集的混合训练（甚至超过一百万条指导），这可能引入数据冗余。为了调查这个问题，我们进行了一系列实证研究，揭示了视觉指导数据集内存在显著冗余，并显示大大减少几个指导数据集的数量甚至不会影响性能。根据研究结果，我们提出了一种新的数据选择方法TIVE，以消除视觉指导数据中的冗余。TIVE首先根据计算的梯度估计视觉指导的任务级和实例级价值。然后，根据估计的价值，TIVE确定了任务级和实例级指导选择策略。

    arXiv:2403.09559v1 Announce Type: new  Abstract: Visual instruction tuning is the key to building multimodal large language models (MLLMs), which greatly improves the reasoning capabilities of large language models (LLMs) in vision scenario. However, existing MLLMs mostly rely on a mixture of multiple highly diverse visual instruction datasets for training (even more than a million instructions), which may introduce data redundancy. To investigate this issue, we conduct a series of empirical studies, which reveal a significant redundancy within the visual instruction datasets, and show that greatly reducing the amount of several instruction dataset even do not affect the performance. Based on the findings, we propose a new data selection approach TIVE, to eliminate redundancy within visual instruction data. TIVE first estimates the task-level and instance-level value of the visual instructions based on computed gradients. Then, according to the estimated values, TIVE determines the tas
    
[^3]: TV-TREES：用于神经符号视频推理的多模态蕴涵树

    TV-TREES: Multimodal Entailment Trees for Neuro-Symbolic Video Reasoning

    [https://arxiv.org/abs/2402.19467](https://arxiv.org/abs/2402.19467)

    TV-TREES是第一个多模态蕴涵树生成器，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树，实现了可解释联合模态推理，并在挑战性的TVQA数据集上展示了最先进的零-shot性能。

    

    在处理电视剪辑等复杂的多模态内容进行问答是一项具有挑战性的任务。这部分是因为当前的视频-语言模型依赖于单模态推理，在处理长输入时性能下降，并且缺乏可解释性。我们提出了TV-TREES，这是第一个多模态蕴涵树生成器。TV-TREES作为一种促进可解释联合模态推理的视频理解方法，通过生成视频直接蕴涵的简单前提与高级结论之间的蕴涵关系树。随后，我们引入了多模态蕴涵树生成任务来评估此类方法的推理质量。我们的方法在具有挑战性的TVQA数据集上的实验结果展示了可解释的、具有最先进零-shot性能的完整视频剪辑，展示了与黑盒方法相比的最佳实践。

    arXiv:2402.19467v1 Announce Type: cross  Abstract: It is challenging to perform question-answering over complex, multimodal content such as television clips. This is in part because current video-language models rely on single-modality reasoning, have lowered performance on long inputs, and lack interpetability. We propose TV-TREES, the first multimodal entailment tree generator. TV-TREES serves as an approach to video understanding that promotes interpretable joint-modality reasoning by producing trees of entailment relationships between simple premises directly entailed by the videos and higher-level conclusions. We then introduce the task of multimodal entailment tree generation to evaluate the reasoning quality of such methods. Our method's experimental results on the challenging TVQA dataset demonstrate intepretable, state-of-the-art zero-shot performance on full video clips, illustrating a best of both worlds contrast to black-box methods.
    
[^4]: SelfFed: 自监督的联邦学习用于IoMT中的数据异质性和标签匮乏问题

    SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT. (arXiv:2307.01514v1 [cs.LG])

    [http://arxiv.org/abs/2307.01514](http://arxiv.org/abs/2307.01514)

    这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。

    

    基于自监督学习的联邦学习范式在行业和研究领域中引起了很大的兴趣，因为它可以协作学习未标记但孤立的数据。然而，自监督的联邦学习策略在标签稀缺和数据异质性（即数据分布不同）方面存在性能下降的问题。在本文中，我们提出了适用于医疗物联网（IoMT）的SelfFed框架。我们的SelfFed框架分为两个阶段。第一个阶段是预训练范式，使用基于Swin Transformer的编码器以分散的方式进行增强建模。SelfFed框架的第一个阶段有助于克服数据异质性问题。第二个阶段是微调范式，引入对比网络和一种在有限标记数据上进行训练的新型聚合策略，用于目标任务的分散训练。这个微调阶段克服了标签稀缺问题。

    Self-supervised learning in federated learning paradigm has been gaining a lot of interest both in industry and research due to the collaborative learning capability on unlabeled yet isolated data. However, self-supervised based federated learning strategies suffer from performance degradation due to label scarcity and diverse data distributions, i.e., data heterogeneity. In this paper, we propose the SelfFed framework for Internet of Medical Things (IoMT). Our proposed SelfFed framework works in two phases. The first phase is the pre-training paradigm that performs augmentive modeling using Swin Transformer based encoder in a decentralized manner. The first phase of SelfFed framework helps to overcome the data heterogeneity issue. The second phase is the fine-tuning paradigm that introduces contrastive network and a novel aggregation strategy that is trained on limited labeled data for a target task in a decentralized manner. This fine-tuning stage overcomes the label scarcity problem
    
[^5]: OpenDriver: 一份开放路况驾驶员状态检测数据集

    OpenDriver: an open-road driver state detection dataset. (arXiv:2304.04203v1 [cs.AI])

    [http://arxiv.org/abs/2304.04203](http://arxiv.org/abs/2304.04203)

    OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。

    

    在现代社会中，道路安全严重依赖于驾驶员的心理和生理状态。疲劳、昏昏欲睡和压力等负面因素会影响驾驶员的反应时间和决策能力，导致交通事故的发生率增加。在众多的驾驶员行为监测研究中，可穿戴生理测量是一种实时监测驾驶员状态的方法。然而，目前在开放道路场景下，缺少驾驶员生理数据集，已有的数据集存在信号质量差、样本量小和数据收集时间短等问题。因此，本文设计并描述了一种大规模多模态驾驶数据集，用于驾驶员受损检测和生物识别数据识别。该数据集包含两种驾驶信号模态：六轴惯性信号和心电图（ECG）信号，这些信号是在100多名驾驶员遵循相同路线行驶时记录的。

    In modern society, road safety relies heavily on the psychological and physiological state of drivers. Negative factors such as fatigue, drowsiness, and stress can impair drivers' reaction time and decision making abilities, leading to an increased incidence of traffic accidents. Among the numerous studies for impaired driving detection, wearable physiological measurement is a real-time approach to monitoring a driver's state. However, currently, there are few driver physiological datasets in open road scenarios and the existing datasets suffer from issues such as poor signal quality, small sample sizes, and short data collection periods. Therefore, in this paper, a large-scale multimodal driving dataset for driver impairment detection and biometric data recognition is designed and described. The dataset contains two modalities of driving signals: six-axis inertial signals and electrocardiogram (ECG) signals, which were recorded while over one hundred drivers were following the same ro
    
[^6]: DualStreamFoveaNet: 一种具有解剖意识的双流融合架构用于鲁棒的中央凹点定位

    DualStreamFoveaNet: A Dual Stream Fusion Architecture with Anatomical Awareness for Robust Fovea Localization. (arXiv:2302.06961v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.06961](http://arxiv.org/abs/2302.06961)

    DualStreamFoveaNet是一种具有解剖意识的双流融合架构，通过利用视网膜和血管分布进行多线索融合，实现对鲁棒的中央凹点定位。实验证明该架构在中央凹点定位方面达到了最先进的性能。

    

    准确的中央凹点定位对于分析视网膜疾病以预防不可逆视力损失至关重要。当前基于深度学习的方法虽然优于传统方法，但仍面临一些挑战，如中央凹点周围局部解剖标记的缺失、不能鲁棒地处理病变视网膜图像和图像条件的变化。本文提出了一种新颖的基于Transformer的架构称为DualStreamFoveaNet (DSFN)用于多线索融合。该架构明确地利用视网膜和血管分布来实现长程连接和全局特征的融合，实现鲁棒的中央凹点定位。我们在双流编码器中引入了一种空间注意机制，用于提取和融合自学习的解剖信息，更注重分布在血管沿线的特征，并通过减少令牌数量显著降低计算成本。我们的广泛实验结果表明，所提出的架构达到了最先进的性能。

    Accurate fovea localization is essential for analyzing retinal diseases to prevent irreversible vision loss. While current deep learning-based methods outperform traditional ones, they still face challenges such as the lack of local anatomical landmarks around the fovea, the inability to robustly handle diseased retinal images, and the variations in image conditions. In this paper, we propose a novel transformer-based architecture called DualStreamFoveaNet (DSFN) for multi-cue fusion. This architecture explicitly incorporates long-range connections and global features using retina and vessel distributions for robust fovea localization. We introduce a spatial attention mechanism in the dual-stream encoder to extract and fuse self-learned anatomical information, focusing more on features distributed along blood vessels and significantly reducing computational costs by decreasing token numbers. Our extensive experiments show that the proposed architecture achieves state-of-the-art perform
    

