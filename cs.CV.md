# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DRCT: Saving Image Super-resolution away from Information Bottleneck](https://arxiv.org/abs/2404.00722) | 基于Vision Transformer的DRCT方法采用创新的机制解决了图像超分辨率中空间信息衰减的问题，提升了模型性能。 |
| [^2] | [ZigMa: Zigzag Mamba Diffusion Model](https://arxiv.org/abs/2403.13802) | 本研究提出了一种名为Zigzag Mamba的零参数方法，通过纠正当前Mamba-based视觉方法中对空间连续性的忽视，实现了更好的速度和内存利用，同时在大分辨率视觉数据集上展示了出色的性能。 |
| [^3] | [ContourDiff: Unpaired Image Translation with Contour-Guided Diffusion Models](https://arxiv.org/abs/2403.10786) | ContourDiff是一种新颖的框架，利用图像的领域不变解剖轮廓表示，旨在帮助准确翻译医学图像并保持其解剖准确性。 |
| [^4] | [HanDiffuser: Text-to-Image Generation With Realistic Hand Appearances](https://arxiv.org/abs/2403.01693) | HanDiffuser提出了一种基于扩散的架构，通过在生成过程中注入手部嵌入来实现逼真的手部外观，包括Text-to-Hand-Params扩散模型和Text-Guided Hand-Params-to-Image扩散模型。 |
| [^5] | [Lifelong Benchmarks: Efficient Model Evaluation in an Era of Rapid Progress](https://arxiv.org/abs/2402.19472) | 提出了终身基准的概念，通过创建不断扩展的大规模基准来减少过拟合风险，并引入了高效的评估框架Sort \& Search（S&S）来解决评估成本问题。 |
| [^6] | [Flexible Physical Camouflage Generation Based on a Differential Approach](https://arxiv.org/abs/2402.13575) | 该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。 |
| [^7] | [Understanding Neural Network Systems for Image Analysis using Vector Spaces and Inverse Maps](https://arxiv.org/abs/2402.00261) | 本文使用线性代数技术将神经网络层视为信号空间之间的映射，并引入了可逆网络的概念和计算产生特定输出的输入图像的算法。 |
| [^8] | [EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](https://arxiv.org/abs/2312.02256) | 提出了高效动态扩散模型（EMDM），能够在更少的采样步骤中实现快速且高质量的动作生成 |
| [^9] | [Efficient Out-of-Distribution Detection with Prototypical Semi-Supervised Learning and Foundation Models](https://arxiv.org/abs/2311.17093) | 本文介绍了一种新的改进的半监督学习方法，利用冻结的基础模型作为神经网络骨干，在半监督学习和超出分布检测方面取得了优越的表现，并引入了新的预训练技术、损失函数和原型选择方法。 |
| [^10] | [Word4Per: Zero-shot Composed Person Retrieval](https://arxiv.org/abs/2311.16515) | 提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。 |
| [^11] | [Octavius: Mitigating Task Interference in MLLMs via MoE](https://arxiv.org/abs/2311.02684) | 提出了一个名为Octavius的新框架，通过结合MoE和LoRA技术设计了一种新颖的LLM解码器LoRA-MoE，用于多模态学习，实验证明其在各种2D和3D下游任务中具有约20%的改进效果。 |
| [^12] | [LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery.](http://arxiv.org/abs/2311.02058) | LOTUS是一种持续模仿学习算法，通过无监督技能发现，使得机器人能够在其整个寿命中持续学习解决新的操作任务。该算法通过构建技能库，并使用元控制器灵活组合技能来提高成功率，在实验中表现出优越的知识传递能力。 |
| [^13] | [FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus.](http://arxiv.org/abs/2310.11178) | FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。 |
| [^14] | [Auto-Prompting SAM for Mobile Friendly 3D Medical Image Segmentation.](http://arxiv.org/abs/2308.14936) | 这项工作提出了一种名为AutoSAM Adapter的方法，用于解决SAM在3D医学图像分割任务上的性能问题。通过参数高效的适应技术，实现了自动提示学习范式，消除了对手动生成提示的需求。 |
| [^15] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |

# 详细

[^1]: DRCT：将图像超分辨率保存在信息瓶颈之外

    DRCT: Saving Image Super-resolution away from Information Bottleneck

    [https://arxiv.org/abs/2404.00722](https://arxiv.org/abs/2404.00722)

    基于Vision Transformer的DRCT方法采用创新的机制解决了图像超分辨率中空间信息衰减的问题，提升了模型性能。

    

    近年来，基于Vision Transformer的低层视觉任务应用取得了广泛的成功。与基于CNN的模型不同，Transformer更擅长捕捉长距离依赖关系，可以利用非局部区域的信息重建图像。在超分辨率领域，基于Swin Transformer的方法已经成为主流，因为它们能够捕捉全局空间信息，并且具有旋转窗口注意机制，有助于在不同窗口之间交换信息。许多研究人员通过扩大感知野或设计复杂网络来提高图像质量和网络效率，取得了令人称赞的结果。然而，我们观察到在前向传播过程中，由于深度增加，空间信息往往会减少，从而导致空间信息的丢失，并最终限制了模型的潜力。

    arXiv:2404.00722v1 Announce Type: cross  Abstract: In recent years, Vision Transformer-based applications to low-level vision tasks have achieved widespread success. Unlike CNN-based models, Transformers are more adept at capturing long-range dependencies, enabling the reconstruction of images utilizing information from non-local areas. In the domain of super-resolution, Swin-transformer-based approaches have become mainstream due to their capacity to capture global spatial information and their shifting-window attention mechanism that facilitates the interchange of information between different windows. Many researchers have enhanced image quality and network efficiency by expanding the receptive field or designing complex networks, yielding commendable results. However, we observed that spatial information tends to diminish during the forward propagation process due to increased depth, leading to a loss of spatial information and, consequently, limiting the model's potential. To addr
    
[^2]: ZigMa：蜿蜒曼巴扩散模型

    ZigMa: Zigzag Mamba Diffusion Model

    [https://arxiv.org/abs/2403.13802](https://arxiv.org/abs/2403.13802)

    本研究提出了一种名为Zigzag Mamba的零参数方法，通过纠正当前Mamba-based视觉方法中对空间连续性的忽视，实现了更好的速度和内存利用，同时在大分辨率视觉数据集上展示了出色的性能。

    

    扩散模型长期以来一直受到可伸缩性和二次复杂性问题的困扰，特别是在基于变压器的结构内部。在这项研究中，我们旨在利用一种称为曼巴的状态空间模型的长序列建模能力，以扩展其在视觉数据生成中的适用性。首先，我们确定了大多数当前基于曼巴的视觉方法中的一个关键疏忽，即曼巴的扫描方案中缺乏对空间连续性的考虑。其次，基于这一洞察力，我们介绍了一种名为Zigzag Mamba的简单、即插即用、零参数方法，它优于基于曼巴的基线，并表现出比基于变压器的基线更快速和更好的内存利用。最后，我们将Zigzag Mamba集成到随机插值框架中，以研究模型在大分辨率视觉数据集（例如FacesHQ $1024\times 1024$和UCF101，MultiModal-CelebA-HQ）上的可伸缩性。

    arXiv:2403.13802v1 Announce Type: cross  Abstract: The diffusion model has long been plagued by scalability and quadratic complexity issues, especially within transformer-based structures. In this study, we aim to leverage the long sequence modeling capability of a State-Space Model called Mamba to extend its applicability to visual data generation. Firstly, we identify a critical oversight in most current Mamba-based vision methods, namely the lack of consideration for spatial continuity in the scan scheme of Mamba. Secondly, building upon this insight, we introduce a simple, plug-and-play, zero-parameter method named Zigzag Mamba, which outperforms Mamba-based baselines and demonstrates improved speed and memory utilization compared to transformer-based baselines. Lastly, we integrate Zigzag Mamba with the Stochastic Interpolant framework to investigate the scalability of the model on large-resolution visual datasets, such as FacesHQ $1024\times 1024$ and UCF101, MultiModal-CelebA-HQ
    
[^3]: ContourDiff：带轮廓引导扩散模型的无配对图像翻译

    ContourDiff: Unpaired Image Translation with Contour-Guided Diffusion Models

    [https://arxiv.org/abs/2403.10786](https://arxiv.org/abs/2403.10786)

    ContourDiff是一种新颖的框架，利用图像的领域不变解剖轮廓表示，旨在帮助准确翻译医学图像并保持其解剖准确性。

    

    准确地在不同模态之间翻译医学图像（例如从CT到MRI）对于许多临床和机器学习应用至关重要。本文提出了一种名为ContourDiff的新框架，该框架利用图像的领域不变解剖轮廓表示。这些表示易于从图像中提取，但对其解剖内容形成精确的空间约束。我们引入一种扩散模型，将来自任意输入领域的图像的轮廓表示转换为输出领域中的图像。

    arXiv:2403.10786v1 Announce Type: cross  Abstract: Accurately translating medical images across different modalities (e.g., CT to MRI) has numerous downstream clinical and machine learning applications. While several methods have been proposed to achieve this, they often prioritize perceptual quality with respect to output domain features over preserving anatomical fidelity. However, maintaining anatomy during translation is essential for many tasks, e.g., when leveraging masks from the input domain to develop a segmentation model with images translated to the output domain. To address these challenges, we propose ContourDiff, a novel framework that leverages domain-invariant anatomical contour representations of images. These representations are simple to extract from images, yet form precise spatial constraints on their anatomical content. We introduce a diffusion model that converts contour representations of images from arbitrary input domains into images in the output domain of in
    
[^4]: HanDiffuser: 具有逼真手部外观的文本图像生成

    HanDiffuser: Text-to-Image Generation With Realistic Hand Appearances

    [https://arxiv.org/abs/2403.01693](https://arxiv.org/abs/2403.01693)

    HanDiffuser提出了一种基于扩散的架构，通过在生成过程中注入手部嵌入来实现逼真的手部外观，包括Text-to-Hand-Params扩散模型和Text-Guided Hand-Params-to-Image扩散模型。

    

    arXiv:2403.01693v1 公告类型: 交叉 文摘: 文本到图像生成模型可以生成高质量的人类形象，但在生成手部时会失去逼真度。常见的缺陷包括不规则的手部姿势、形状、错误的手指数量以及物理上不合理的手指方向。为了生成具有逼真手部的图像，我们提出了一种基于扩散的新颖架构，称为HanDiffuser，通过在生成过程中注入手部嵌入来实现逼真度。HanDiffuser包括两个组件:Text-to-Hand-Params扩散模型，用于从输入文本提示生成SMPL-身体和MANO-手部参数，以及Text-Guided Hand-Params-to-Image扩散模型，通过在上一部件生成的提示和手部参数上进行调节来合成图像。我们合并了手部表示的多个方面，包括3D形状和关节级手指位置、方向和关节，以实现强大的学习和可靠的推断性能。

    arXiv:2403.01693v1 Announce Type: cross  Abstract: Text-to-image generative models can generate high-quality humans, but realism is lost when generating hands. Common artifacts include irregular hand poses, shapes, incorrect numbers of fingers, and physically implausible finger orientations. To generate images with realistic hands, we propose a novel diffusion-based architecture called HanDiffuser that achieves realism by injecting hand embeddings in the generative process. HanDiffuser consists of two components: a Text-to-Hand-Params diffusion model to generate SMPL-Body and MANO-Hand parameters from input text prompts, and a Text-Guided Hand-Params-to-Image diffusion model to synthesize images by conditioning on the prompts and hand parameters generated by the previous component. We incorporate multiple aspects of hand representation, including 3D shapes and joint-level finger positions, orientations and articulations, for robust learning and reliable performance during inference. We
    
[^5]: 终身基准：在快速进展时代中高效的模型评估

    Lifelong Benchmarks: Efficient Model Evaluation in an Era of Rapid Progress

    [https://arxiv.org/abs/2402.19472](https://arxiv.org/abs/2402.19472)

    提出了终身基准的概念，通过创建不断扩展的大规模基准来减少过拟合风险，并引入了高效的评估框架Sort \& Search（S&S）来解决评估成本问题。

    

    标准化基准推动机器学习的进步。然而，通过重复测试，算法对基准的特殊性过度利用，会增加过拟合的风险。在我们的工作中，我们试图通过编制不断扩展的大规模基准（称为终身基准）来缓解这一挑战。作为我们方法的示例，我们创建了终身-CIFAR10和终身-ImageNet，分别包含（目前）1.69百万和1.98百万个测试样本。尽管减少了过拟合，终身基准引入了一个关键挑战：评估日益增多的模型在不断扩大的样本集上的高成本。为了解决这一挑战，我们还引入了一种高效的评估框架：Sort \& Search (S&S)，通过利用动态规划算法有选择地对测试样本进行排序和子选择，使得终身基准评估具有成本效益。通过对31,000个模型进行广泛的实证评估

    arXiv:2402.19472v1 Announce Type: new  Abstract: Standardized benchmarks drive progress in machine learning. However, with repeated testing, the risk of overfitting grows as algorithms over-exploit benchmark idiosyncrasies. In our work, we seek to mitigate this challenge by compiling ever-expanding large-scale benchmarks called Lifelong Benchmarks. As exemplars of our approach, we create Lifelong-CIFAR10 and Lifelong-ImageNet, containing (for now) 1.69M and 1.98M test samples, respectively. While reducing overfitting, lifelong benchmarks introduce a key challenge: the high cost of evaluating a growing number of models across an ever-expanding sample set. To address this challenge, we also introduce an efficient evaluation framework: Sort \& Search (S&S), which reuses previously evaluated models by leveraging dynamic programming algorithms to selectively rank and sub-select test samples, enabling cost-effective lifelong benchmarking. Extensive empirical evaluations across 31,000 models 
    
[^6]: 基于差异方法的灵活物理伪装生成

    Flexible Physical Camouflage Generation Based on a Differential Approach

    [https://arxiv.org/abs/2402.13575](https://arxiv.org/abs/2402.13575)

    该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。

    

    这项研究介绍了一种新的神经渲染方法，专门针对对抗伪装，在广泛的三维渲染框架内进行了定制。我们的方法，名为FPA，通过忠实地模拟光照条件和材料变化，确保在三维目标上对纹理进行微妙而逼真的表现。为了实现这一目标，我们采用一种生成方法，从扩散模型中学习对抗模式。这涉及将一个特别设计的对抗损失和隐蔽约束损失结合在一起，以确保伪装在物理世界中的对抗性和隐蔽性质。此外，我们展示了所提出的伪装在贴纸模式下的有效性，展示了其覆盖目标而不影响对抗信息的能力。通过实证和物理实验，FPA在攻击成功率和可转移性方面表现出很强的性能。

    arXiv:2402.13575v1 Announce Type: cross  Abstract: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferabili
    
[^7]: 使用向量空间和逆映射了解图像分析中神经网络系统的研究

    Understanding Neural Network Systems for Image Analysis using Vector Spaces and Inverse Maps

    [https://arxiv.org/abs/2402.00261](https://arxiv.org/abs/2402.00261)

    本文使用线性代数技术将神经网络层视为信号空间之间的映射，并引入了可逆网络的概念和计算产生特定输出的输入图像的算法。

    

    开发数学方法来理解图像分析中复杂的神经网络系统具有极大的兴趣。本文介绍了利用线性代数技术将神经网络层视为信号空间之间的映射的方法。首先，我们演示了如何使用信号空间来可视化权重空间和卷积层卷积核。其次，我们引入了可逆网络的概念和计算产生特定输出的输入图像的算法。我们在两个可逆网络和ResNet18上演示了我们的方法。

    There is strong interest in developing mathematical methods that can be used to understand complex neural networks used in image analysis. In this paper, we introduce techniques from Linear Algebra to model neural network layers as maps between signal spaces. First, we demonstrate how signal spaces can be used to visualize weight spaces and convolutional layer kernels. We also demonstrate how residual vector spaces can be used to further visualize information lost at each layer. Second, we introduce the concept of invertible networks and an algorithm for computing input images that yield specific outputs. We demonstrate our approach on two invertible networks and ResNet18.
    
[^8]: 高效动态扩散模型（EMDM）用于快速且高质量的动作生成

    EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation

    [https://arxiv.org/abs/2312.02256](https://arxiv.org/abs/2312.02256)

    提出了高效动态扩散模型（EMDM），能够在更少的采样步骤中实现快速且高质量的动作生成

    

    我们引入了高效的动态扩散模型（EMDM），用于快速且高质量的人类动作生成。当前最先进的生成式扩散模型取得了令人印象深刻的结果，但往往在追求快速生成的同时牺牲了质量。为了解决这些问题，我们提出了EMDM，它通过在扩散模型中的多次采样步骤中捕捉复杂分布，实现了更少的采样步骤和生成过程的显着加速。

    arXiv:2312.02256v2 Announce Type: replace-cross  Abstract: We introduce Efficient Motion Diffusion Model (EMDM) for fast and high-quality human motion generation. Current state-of-the-art generative diffusion models have produced impressive results but struggle to achieve fast generation without sacrificing quality. On the one hand, previous works, like motion latent diffusion, conduct diffusion within a latent space for efficiency, but learning such a latent space can be a non-trivial effort. On the other hand, accelerating generation by naively increasing the sampling step size, e.g., DDIM, often leads to quality degradation as it fails to approximate the complex denoising distribution. To address these issues, we propose EMDM, which captures the complex distribution during multiple sampling steps in the diffusion model, allowing for much fewer sampling steps and significant acceleration in generation. This is achieved by a conditional denoising diffusion GAN to capture multimodal da
    
[^9]: 用原型半监督学习和基础模型实现高效的超出分布检测

    Efficient Out-of-Distribution Detection with Prototypical Semi-Supervised Learning and Foundation Models

    [https://arxiv.org/abs/2311.17093](https://arxiv.org/abs/2311.17093)

    本文介绍了一种新的改进的半监督学习方法，利用冻结的基础模型作为神经网络骨干，在半监督学习和超出分布检测方面取得了优越的表现，并引入了新的预训练技术、损失函数和原型选择方法。

    

    本文介绍了PAWS-VMK，一种改进的原型半监督学习方法，专门设计用于利用冻结的基础模型作为神经网络骨干，该方法在计算机视觉领域中优于以往的半监督学习和超出分布（OOD）检测结果，改进了Predicting View-Assignments With Support Samples（PAWS）半监督学习方法。我们引入了(1) 参数化von-Mises Fisher随机邻域嵌入（vMF-SNE）来预训练投影头，使用基础模型的高质量嵌入;(2) 受MixMatch启发的损失，通过对多视图的预测进行平均，提供比PAWS中使用的一致性损失更可靠的监督信号;和(3) 简单k-Means原型选择（SKMPS），一种比其他无监督标签选择方法提供更优越性能的技术。

    arXiv:2311.17093v2 Announce Type: replace-cross  Abstract: This paper describes PAWS-VMK, an improved approach to prototypical semi-supervised learning in the field of computer vision, specifically designed to utilize a frozen foundation model as the neural network backbone. This method outperforms previous results in semi-supervised learning and out-of-distribution (OOD) detection, improving upon the Predicting View-Assignments With Support Samples (PAWS) semi-supervised learning method. We introduce (1) parametric von-Mises Fisher Stochastic Neighbour Embedding (vMF-SNE) to pretrain the projection head using the high-quality embeddings of the foundation model; (2) a MixMatch inspired loss, where predictions across multiple views are averaged to provide a more reliable supervision signal compared to the consistency loss used in PAWS and (3) simple $k$-Means prototype selection (SKMPS), a technique that provides superior performance to other unsupervised label selection approaches in t
    
[^10]: Word4Per: Zero-shot组合人员检索

    Word4Per: Zero-shot Composed Person Retrieval

    [https://arxiv.org/abs/2311.16515](https://arxiv.org/abs/2311.16515)

    提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。

    

    寻找特定人员具有极大的社会效益和安全价值，通常涉及视觉和文本信息的结合。本文提出了一个全新的任务，称为组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索。然而，监督CPR需要昂贵的手动注释数据集，而目前没有可用资源。为了解决这个问题，我们首先引入了零样本组合人员检索（ZS-CPR），利用现有的领域相关数据解决了CPR问题而不需要昂贵的注释。其次，为了学习ZS-CPR模型，我们提出了一个两阶段学习框架，即Word4Per，其中包含一个轻量级的文本反转网络。

    arXiv:2311.16515v2 Announce Type: replace-cross  Abstract: Searching for specific person has great social benefits and security value, and it often involves a combination of visual and textual information. Conventional person retrieval methods, whether image-based or text-based, usually fall short in effectively harnessing both types of information, leading to the loss of accuracy. In this paper, a whole new task called Composed Person Retrieval (CPR) is proposed to jointly utilize both image and text information for target person retrieval. However, the supervised CPR requires very costly manual annotation dataset, while there are currently no available resources. To mitigate this issue, we firstly introduce the Zero-shot Composed Person Retrieval (ZS-CPR), which leverages existing domain-related data to resolve the CPR problem without expensive annotations. Secondly, to learn ZS-CPR model, we propose a two-stage learning framework, Word4Per, where a lightweight Textual Inversion Netw
    
[^11]: Octavius：通过MoE减轻MLLM中的任务干扰

    Octavius: Mitigating Task Interference in MLLMs via MoE

    [https://arxiv.org/abs/2311.02684](https://arxiv.org/abs/2311.02684)

    提出了一个名为Octavius的新框架，通过结合MoE和LoRA技术设计了一种新颖的LLM解码器LoRA-MoE，用于多模态学习，实验证明其在各种2D和3D下游任务中具有约20%的改进效果。

    

    最近的研究表明，大型语言模型（LLMs）可以通过指导调整将它们的零-shot泛化能力扩展到多模态学习。随着引入更多的形式和下游任务，负面冲突和干扰可能对性能产生更严重的影响。虽然这种现象在以前的工作中被忽视了，但我们提出了一个名为\mname 的新颖且可扩展的框架，用于与Multimodal Large Language Models（MLLMs）一起进行多模态学习的全面研究和实验。具体来说，我们结合了众所周知的专家混合（MoE）和代表性PEFT技术之一，即LoRA，设计了一种新颖的基于LLM的解码器，称为LoRA-MoE，用于多模态学习。实验结果（约20\%的改进）表明了我们设计在各种2D和3D下游任务中的有效性和多功能性。代码和相应数据集将很快提供。

    arXiv:2311.02684v1 Announce Type: cross  Abstract: Recent studies have demonstrated Large Language Models (LLMs) can extend their zero-shot generalization capabilities to multimodal learning through instruction tuning. As more modalities and downstream tasks are introduced, negative conflicts and interference may have a worse impact on performance. While this phenomenon has been overlooked in previous work, we propose a novel and extensible framework, called \mname, for comprehensive studies and experimentation on multimodal learning with Multimodal Large Language Models (MLLMs). Specifically, we combine the well-known Mixture-of-Experts (MoE) and one of the representative PEFT techniques, \emph{i.e.,} LoRA, designing a novel LLM-based decoder, called LoRA-MoE, for multimodal learning. The experimental results (about 20\% improvement) have shown the effectiveness and versatility of our design in various 2D and 3D downstream tasks. Code and corresponding dataset will be available soon.
    
[^12]: LOTUS：通过无监督技能发现的持续模仿学习，用于机器人操作

    LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery. (arXiv:2311.02058v1 [cs.RO])

    [http://arxiv.org/abs/2311.02058](http://arxiv.org/abs/2311.02058)

    LOTUS是一种持续模仿学习算法，通过无监督技能发现，使得机器人能够在其整个寿命中持续学习解决新的操作任务。该算法通过构建技能库，并使用元控制器灵活组合技能来提高成功率，在实验中表现出优越的知识传递能力。

    

    我们介绍了一种名为LOTUS的持续模仿学习算法，它使得物理机器人能够在其整个寿命中持续而高效地学习解决新的操作任务。LOTUS的核心思想是通过一系列新任务的少量人类演示构建一个不断增长的技能库。LOTUS首先使用开放词汇视觉模型进行持续技能发现过程，该模型从未分段的演示中提取重复出现的技能模式。持续技能发现更新现有技能以避免对以前任务的灾难性遗忘，并添加新技能以解决新任务。LOTUS训练一个元控制器，在终身学习过程中灵活地组合各种技能来解决基于视觉的操作任务。我们的综合实验证明，与先前方法相比，LOTUS在成功率上超过了现有技术基线方法11％以上，显示了其优越的知识传递能力。

    We introduce LOTUS, a continual imitation learning algorithm that empowers a physical robot to continuously and efficiently learn to solve new manipulation tasks throughout its lifespan. The core idea behind LOTUS is constructing an ever-growing skill library from a sequence of new tasks with a small number of human demonstrations. LOTUS starts with a continual skill discovery process using an open-vocabulary vision model, which extracts skills as recurring patterns presented in unsegmented demonstrations. Continual skill discovery updates existing skills to avoid catastrophic forgetting of previous tasks and adds new skills to solve novel tasks. LOTUS trains a meta-controller that flexibly composes various skills to tackle vision-based manipulation tasks in the lifelong learning process. Our comprehensive experiments show that LOTUS outperforms state-of-the-art baselines by over 11% in success rate, showing its superior knowledge transfer ability compared to prior methods. More result
    
[^13]: FocDepthFormer: 使用LSTM的Transformer用于从焦点进行深度估计

    FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus. (arXiv:2310.11178v1 [cs.CV])

    [http://arxiv.org/abs/2310.11178](http://arxiv.org/abs/2310.11178)

    FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。

    

    从焦点堆栈进行深度估计是一个基本的计算机视觉问题，旨在通过图像堆栈中的焦点/离焦线索推断深度。大多数现有方法通过在一组固定的图像堆栈上应用二维或三维卷积神经网络（CNNs）来处理此问题，以在图像和堆栈之间学习特征。由于CNN的局部性质，它们的性能受到限制，并且它们被限制在处理在训练和推断中一致的固定数量的堆栈上，从而限制了对任意长度堆栈的泛化能力。为了解决上述限制，我们开发了一种新颖的基于Transformer的网络，FocDepthFormer，主要由带有LSTM模块和CNN解码器的Transformer组成。Transformer中的自注意力通过隐含非局部交叉参考能够学习更多有信息的特征。LSTM模块被学习用于将表示集成到具有任意图像的堆栈中。为了直接捕获低级特征

    Depth estimation from focal stacks is a fundamental computer vision problem that aims to infer depth from focus/defocus cues in the image stacks. Most existing methods tackle this problem by applying convolutional neural networks (CNNs) with 2D or 3D convolutions over a set of fixed stack images to learn features across images and stacks. Their performance is restricted due to the local properties of the CNNs, and they are constrained to process a fixed number of stacks consistent in train and inference, limiting the generalization to the arbitrary length of stacks. To handle the above limitations, we develop a novel Transformer-based network, FocDepthFormer, composed mainly of a Transformer with an LSTM module and a CNN decoder. The self-attention in Transformer enables learning more informative features via an implicit non-local cross reference. The LSTM module is learned to integrate the representations across the stack with arbitrary images. To directly capture the low-level featur
    
[^14]: 为移动友好的3D医学图像分割自动提示SAM

    Auto-Prompting SAM for Mobile Friendly 3D Medical Image Segmentation. (arXiv:2308.14936v1 [cs.CV])

    [http://arxiv.org/abs/2308.14936](http://arxiv.org/abs/2308.14936)

    这项工作提出了一种名为AutoSAM Adapter的方法，用于解决SAM在3D医学图像分割任务上的性能问题。通过参数高效的适应技术，实现了自动提示学习范式，消除了对手动生成提示的需求。

    

    Segment Anything Model (SAM)已经被迅速应用于各种自然图像的分割。然而，最近的研究表明，SAM在3D医学图像分割任务上的性能不佳。除了自然图像和医学图像之间的领域差距外，2D和3D图像之间的空间布局差异，强大的GPU服务器所带来的大量计算负担，以及耗时的手动提示生成使得SAM无法扩展到更广泛的医学图像分割应用。为了解决这些挑战，在这项工作中，我们引入了一种新方法AutoSAM Adapter，专为3D多器官CT分割而设计。我们采用参数高效的适应技术开发了一种自动提示学习范式，以促进将SAM模型的能力转化为3D医学图像分割，消除了手动生成提示的需求。

    The Segment Anything Model (SAM) has rapidly been adopted for segmenting a wide range of natural images. However, recent studies have indicated that SAM exhibits subpar performance on 3D medical image segmentation tasks. In addition to the domain gaps between natural and medical images, disparities in the spatial arrangement between 2D and 3D images, the substantial computational burden imposed by powerful GPU servers, and the time-consuming manual prompt generation impede the extension of SAM to a broader spectrum of medical image segmentation applications. To address these challenges, in this work, we introduce a novel method, AutoSAM Adapter, designed specifically for 3D multi-organ CT-based segmentation. We employ parameter-efficient adaptation techniques in developing an automatic prompt learning paradigm to facilitate the transformation of the SAM model's capabilities to 3D medical image segmentation, eliminating the need for manually generated prompts. Furthermore, we effectivel
    
[^15]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    

