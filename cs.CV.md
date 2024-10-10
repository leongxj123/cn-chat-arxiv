# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QKFormer: Hierarchical Spiking Transformer using Q-K Attention](https://arxiv.org/abs/2403.16552) | QKFormer引入了新颖的脉冲形式Q-K注意力机制、分层结构和补丁嵌入模块，以提高脉冲变压器的性能。 |
| [^2] | [Reward Guided Latent Consistency Distillation](https://arxiv.org/abs/2403.11027) | 该论文提出了一种奖励引导的潜在一致性蒸馏方法，通过在LCD过程中整合奖励模型的反馈，从而有效提高高保真图像生成时的样本质量。 |
| [^3] | [Topologically faithful multi-class segmentation in medical images](https://arxiv.org/abs/2403.11001) | 提出了一种用于医学图像的拓扑保真多类别分割的通用损失函数，通过将N类别分割问题分解为N个单类别分割任务，实现了对神经网络的训练，验证了在四个医学数据集上的有效性 |
| [^4] | [Less is More: Data Value Estimation for Visual Instruction Tuning](https://arxiv.org/abs/2403.09559) | 视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。 |
| [^5] | [DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation](https://arxiv.org/abs/2403.07733) | 通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。 |
| [^6] | [ObjectCompose: Evaluating Resilience of Vision-Based Models on Object-to-Background Compositional Changes](https://arxiv.org/abs/2403.04701) | 评估基于视觉的模型对于物体与背景之间多样化变化的鲁棒性，提出一种可以引入不同对象方面变化的方法 |
| [^7] | [RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation](https://arxiv.org/abs/2402.15487) | 本文提出了交互式场景探索任务，通过自主探索环境生成了动作条件化场景图，捕捉了环境的结构 |
| [^8] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^9] | [Feudal Networks for Visual Navigation](https://arxiv.org/abs/2402.12498) | 使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。 |
| [^10] | [LM-HT SNN: Enhancing the Performance of SNN to ANN Counterpart through Learnable Multi-hierarchical Threshold Model](https://arxiv.org/abs/2402.00411) | 本文通过提出LM-HT模型，一个可学习的多层次阈值模型，增强了脉冲神经网络（SNN）与人工神经网络（ANN）的性能对应关系。 |
| [^11] | [Fuse Your Latents: Video Editing with Multi-source Latent Diffusion Models.](http://arxiv.org/abs/2310.16400) | 本论文提出了一种名为FLDM的无需训练的框架，通过融合图像 Latent Diffusion Model（LDM）和视频 LDM，在视频编辑过程中实现了文本引导的视频编辑。这一方法既保持了视频的时间一致性，又利用了图像 LDM 的高保真度，并且具有灵活性与可替换性。 |
| [^12] | [Learning to Grow Artificial Hippocampi in Vision Transformers for Resilient Lifelong Learning.](http://arxiv.org/abs/2303.08250) | 本文提出了一种在Vision Transformer中学习成长人工海马的方法，以实现弹性终身学习。通过神经架构搜索进行维护，选取多头自注意力块中的最终线性投影层进行ArtiHippo的实现和成长。 |

# 详细

[^1]: QKFormer: 使用Q-K注意力的分层脉冲变压器

    QKFormer: Hierarchical Spiking Transformer using Q-K Attention

    [https://arxiv.org/abs/2403.16552](https://arxiv.org/abs/2403.16552)

    QKFormer引入了新颖的脉冲形式Q-K注意力机制、分层结构和补丁嵌入模块，以提高脉冲变压器的性能。

    

    脉冲变压器将脉冲神经网络（SNNs）与变压器架构相结合，由于其节能高性能的潜力，吸引了很多关注。然而，该领域现有模型仍然存在性能不佳的问题。为了提高性能，我们引入了几项创新：i）我们提出了一种为SNNs量身定制的新型脉冲形式Q-K注意力机制，通过具有线性复杂性的二进制向量有效地建模令牌或通道维度的重要性。ii）我们将具有显著性能优势的分层结构引入脉冲变压器，从而获得多尺度脉冲表示，这对大脑和人工神经网络的性能都有显着好处。iii）我们设计了一个通用且强大的补丁嵌入模块，其中包含了一个专门为脉冲变压器设计的变形快捷方式。总之，我们开发了QKFormer，一种分层脉冲变压器。

    arXiv:2403.16552v1 Announce Type: cross  Abstract: Spiking Transformers, which integrate Spiking Neural Networks (SNNs) with Transformer architectures, have attracted significant attention due to their potential for energy efficiency and high performance. However, existing models in this domain still suffer from suboptimal performance. We introduce several innovations to improve the performance: i) We propose a novel spike-form Q-K attention mechanism, tailored for SNNs, which efficiently models the importance of token or channel dimensions through binary vectors with linear complexity. ii) We incorporate the hierarchical structure, which significantly benefits the performance of both the brain and artificial neural networks, into spiking transformers to obtain multi-scale spiking representation. iii) We design a versatile and powerful patch embedding module with a deformed shortcut specifically for spiking transformers. Together, we develop QKFormer, a hierarchical spiking transformer
    
[^2]: 奖励引导的潜在一致性蒸馏

    Reward Guided Latent Consistency Distillation

    [https://arxiv.org/abs/2403.11027](https://arxiv.org/abs/2403.11027)

    该论文提出了一种奖励引导的潜在一致性蒸馏方法，通过在LCD过程中整合奖励模型的反馈，从而有效提高高保真图像生成时的样本质量。

    

    潜在一致性蒸馏(LCD)已成为一种有效的文本到图像合成范式。通过从预训练的教师潜在扩散模型(LDM)中蒸馏出潜在一致性模型(LCM)，LCD在仅需2到4个推理步骤内促进了高保真图像的生成。然而，LCM的高效推理是以样本质量为代价的。本文提出通过在训练过程中将LCM的输出与人类偏好对齐来补偿质量损失。具体而言，我们引入奖励引导的LCD(RG-LCD)，通过将奖励模型(RM)的反馈整合到LCD过程中，通过将原始LCD损失与最大化与LCM单步生成相关联的奖励的目标相结合，来最大化奖励。通过人类评估验证，当使用良好RM的反馈进行训练时，我们的RG-LCM的2步生成被人类青睐，超过了50步DDIM样本。

    arXiv:2403.11027v1 Announce Type: cross  Abstract: Latent Consistency Distillation (LCD) has emerged as a promising paradigm for efficient text-to-image synthesis. By distilling a latent consistency model (LCM) from a pre-trained teacher latent diffusion model (LDM), LCD facilitates the generation of high-fidelity images within merely 2 to 4 inference steps. However, the LCM's efficient inference is obtained at the cost of the sample quality. In this paper, we propose compensating the quality loss by aligning LCM's output with human preference during training. Specifically, we introduce Reward Guided LCD (RG-LCD), which integrates feedback from a reward model (RM) into the LCD process by augmenting the original LCD loss with the objective of maximizing the reward associated with LCM's single-step generation. As validated through human evaluation, when trained with the feedback of a good RM, the 2-step generations from our RG-LCM are favored by humans over the 50-step DDIM samples from 
    
[^3]: 医学图像中拓扑保真的多类别分割

    Topologically faithful multi-class segmentation in medical images

    [https://arxiv.org/abs/2403.11001](https://arxiv.org/abs/2403.11001)

    提出了一种用于医学图像的拓扑保真多类别分割的通用损失函数，通过将N类别分割问题分解为N个单类别分割任务，实现了对神经网络的训练，验证了在四个医学数据集上的有效性

    

    在医学图像分割中，拓扑精度是一个非常重要的属性，对于下游应用如网络分析和血管或细胞计数中的流模拟至关重要。最近，重要的方法论进步将代数拓扑中扎实的概念带到了二值分割中。然而，在多类别分割场景中，这些方法很少被探索，拓扑错误很常见。我们提出了一个通用损失函数，用于拓扑保真的多类别分割，扩展了最近基于持久条码的Betti匹配概念。我们将N类别分割问题投影到N个单类别分割任务，这使得我们能够使用一参数持久同调，从而使神经网络的训练变得可行。我们在一组包含高度不同拓扑特征的四个医学数据集上验证了我们的方法。

    arXiv:2403.11001v1 Announce Type: cross  Abstract: Topological accuracy in medical image segmentation is a highly important property for downstream applications such as network analysis and flow modeling in vessels or cell counting. Recently, significant methodological advancements have brought well-founded concepts from algebraic topology to binary segmentation. However, these approaches have been underexplored in multi-class segmentation scenarios, where topological errors are common. We propose a general loss function for topologically faithful multi-class segmentation extending the recent Betti matching concept, which is based on induced matchings of persistence barcodes. We project the N-class segmentation problem to N single-class segmentation tasks, which allows us to use 1-parameter persistent homology making training of neural networks computationally feasible. We validate our method on a comprehensive set of four medical datasets with highly variant topological characteristic
    
[^4]: 数据价值评估对视觉指导调整的影响

    Less is More: Data Value Estimation for Visual Instruction Tuning

    [https://arxiv.org/abs/2403.09559](https://arxiv.org/abs/2403.09559)

    视觉指导调整时需要进行数据价值评估，通过新的数据选择方法TIVE，根据任务级和实例级价值来消除视觉指导数据中的冗余。

    

    视觉指导调整是构建多模式大语言模型（MLLMs）的关键，大大提高了大语言模型（LLMs）在视觉场景中的推理能力。然而，现有的MLLMs主要依赖于多个高度多样化的视觉指导数据集的混合训练（甚至超过一百万条指导），这可能引入数据冗余。为了调查这个问题，我们进行了一系列实证研究，揭示了视觉指导数据集内存在显著冗余，并显示大大减少几个指导数据集的数量甚至不会影响性能。根据研究结果，我们提出了一种新的数据选择方法TIVE，以消除视觉指导数据中的冗余。TIVE首先根据计算的梯度估计视觉指导的任务级和实例级价值。然后，根据估计的价值，TIVE确定了任务级和实例级指导选择策略。

    arXiv:2403.09559v1 Announce Type: new  Abstract: Visual instruction tuning is the key to building multimodal large language models (MLLMs), which greatly improves the reasoning capabilities of large language models (LLMs) in vision scenario. However, existing MLLMs mostly rely on a mixture of multiple highly diverse visual instruction datasets for training (even more than a million instructions), which may introduce data redundancy. To investigate this issue, we conduct a series of empirical studies, which reveal a significant redundancy within the visual instruction datasets, and show that greatly reducing the amount of several instruction dataset even do not affect the performance. Based on the findings, we propose a new data selection approach TIVE, to eliminate redundancy within visual instruction data. TIVE first estimates the task-level and instance-level value of the visual instructions based on computed gradients. Then, according to the estimated values, TIVE determines the tas
    
[^5]: DSEG-LIME -- 通过层次化数据驱动分割提升图像解释能力

    DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation

    [https://arxiv.org/abs/2403.07733](https://arxiv.org/abs/2403.07733)

    通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。

    

    可解释的人工智能在揭示复杂机器学习模型的决策过程中至关重要。LIME (Local Interpretable Model-agnostic Explanations) 是一个广为人知的用于图像分析的XAI框架。它利用图像分割来创建特征以识别相关的分类区域。然而，较差的分割可能会影响解释的一致性并削弱各个区域的重要性，从而影响整体的可解释性。针对这些挑战，我们引入了DSEG-LIME (Data-Driven Segmentation LIME)，具有: i) 用于生成人类可识别特征的数据驱动分割, 和 ii) 通过组合实现的层次分割程序。我们在预训练模型上使用来自ImageNet数据集的图像对DSEG-LIME进行基准测试-这些情景不包含特定领域的知识。分析包括使用已建立的XAI指标进行定量评估，以及进一步的定性评估。

    arXiv:2403.07733v1 Announce Type: cross  Abstract: Explainable Artificial Intelligence is critical in unraveling decision-making processes in complex machine learning models. LIME (Local Interpretable Model-agnostic Explanations) is a well-known XAI framework for image analysis. It utilizes image segmentation to create features to identify relevant areas for classification. Consequently, poor segmentation can compromise the consistency of the explanation and undermine the importance of the segments, affecting the overall interpretability. Addressing these challenges, we introduce DSEG-LIME (Data-Driven Segmentation LIME), featuring: i) a data-driven segmentation for human-recognized feature generation, and ii) a hierarchical segmentation procedure through composition. We benchmark DSEG-LIME on pre-trained models with images from the ImageNet dataset - scenarios without domain-specific knowledge. The analysis includes a quantitative evaluation using established XAI metrics, complemented
    
[^6]: ObjectCompose: 评估基于视觉的模型在物体与背景组合变化上的韧性

    ObjectCompose: Evaluating Resilience of Vision-Based Models on Object-to-Background Compositional Changes

    [https://arxiv.org/abs/2403.04701](https://arxiv.org/abs/2403.04701)

    评估基于视觉的模型对于物体与背景之间多样化变化的鲁棒性，提出一种可以引入不同对象方面变化的方法

    

    由于最近基于视觉的模型进行了大规模多模态训练并具有泛化能力，了解它们的鲁棒性程度对于它们在现实世界中的部署至关重要。在本研究中，我们评估了当前基于视觉的模型针对不同的物体与背景上下文变化的韧性。大多数鲁棒性评估方法引入了合成数据集来诱导物体特征（视点、尺度、颜色）的变化，或者利用图像转换技术（对抗性变化、常见破坏）在真实图像上模拟分布的变化。最近的研究探索了利用大语言模型和扩散模型来生成背景的变化。但是，这些方法要么在提供对要进行的更改的控制方面不足，要么扭曲了物体的语义，使其不适用于任务。与之相反，我们的方法可以引入各种对象

    arXiv:2403.04701v1 Announce Type: cross  Abstract: Given the large-scale multi-modal training of recent vision-based models and their generalization capabilities, understanding the extent of their robustness is critical for their real-world deployment. In this work, we evaluate the resilience of current vision-based models against diverse object-to-background context variations. The majority of robustness evaluation methods have introduced synthetic datasets to induce changes to object characteristics (viewpoints, scale, color) or utilized image transformation techniques (adversarial changes, common corruptions) on real images to simulate shifts in distributions. Recent works have explored leveraging large language models and diffusion models to generate changes in the background. However, these methods either lack in offering control over the changes to be made or distort the object semantics, making them unsuitable for the task. Our method, on the other hand, can induce diverse objec
    
[^7]: RoboEXP: 通过交互式探索实现动作条件化场景图用于机器人操作

    RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation

    [https://arxiv.org/abs/2402.15487](https://arxiv.org/abs/2402.15487)

    本文提出了交互式场景探索任务，通过自主探索环境生成了动作条件化场景图，捕捉了环境的结构

    

    机器人需要探索周围环境以适应并应对未知环境中的任务。本文介绍了交互式场景探索的新任务，其中机器人自主探索环境并生成一个捕捉基础环境结构的动作条件化场景图（ACSG）

    arXiv:2402.15487v1 Announce Type: cross  Abstract: Robots need to explore their surroundings to adapt to and tackle tasks in unknown environments. Prior work has proposed building scene graphs of the environment but typically assumes that the environment is static, omitting regions that require active interactions. This severely limits their ability to handle more complex tasks in household and office environments: before setting up a table, robots must explore drawers and cabinets to locate all utensils and condiments. In this work, we introduce the novel task of interactive scene exploration, wherein robots autonomously explore environments and produce an action-conditioned scene graph (ACSG) that captures the structure of the underlying environment. The ACSG accounts for both low-level information, such as geometry and semantics, and high-level information, such as the action-conditioned relationships between different entities in the scene. To this end, we present the Robotic Explo
    
[^8]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^9]: 封建网络用于视觉导航

    Feudal Networks for Visual Navigation

    [https://arxiv.org/abs/2402.12498](https://arxiv.org/abs/2402.12498)

    使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。

    

    视觉导航遵循人类可以在没有详细地图的情况下导航的直觉。一种常见方法是在建立包含可用于规划的图像节点的拓扑图的同时进行交互式探索。最近的变体从被动视频中学习，并可以利用复杂的社交和语义线索进行导航。然而，需要大量的训练视频，利用大型图并且由于使用了里程计，场景不是未知的。我们引入了一种使用封建学习的视觉导航的新方法，该方法采用了由工作代理、中级管理者和高级管理者组成的分层结构。封建学习范式的关键在于，每个级别的代理看到任务的不同方面，并且在不同的空间和时间尺度上运作。在此框架中开发了两个独特的模块。对于高级管理者，我们自监督地学习一个记忆代理地图以记录

    arXiv:2402.12498v1 Announce Type: cross  Abstract: Visual navigation follows the intuition that humans can navigate without detailed maps. A common approach is interactive exploration while building a topological graph with images at nodes that can be used for planning. Recent variations learn from passive videos and can navigate using complex social and semantic cues. However, a significant number of training videos are needed, large graphs are utilized, and scenes are not unseen since odometry is utilized. We introduce a new approach to visual navigation using feudal learning, which employs a hierarchical structure consisting of a worker agent, a mid-level manager, and a high-level manager. Key to the feudal learning paradigm, agents at each level see a different aspect of the task and operate at different spatial and temporal scales. Two unique modules are developed in this framework. For the high- level manager, we learn a memory proxy map in a self supervised manner to record prio
    
[^10]: LM-HT SNN: 通过可学习的多层次阈值模型增强SNN与ANN的性能对应关系

    LM-HT SNN: Enhancing the Performance of SNN to ANN Counterpart through Learnable Multi-hierarchical Threshold Model

    [https://arxiv.org/abs/2402.00411](https://arxiv.org/abs/2402.00411)

    本文通过提出LM-HT模型，一个可学习的多层次阈值模型，增强了脉冲神经网络（SNN）与人工神经网络（ANN）的性能对应关系。

    

    与传统的人工神经网络（ANN）相比，脉冲神经网络（SNN）因其更具生物启发和能量效率的信息传递能力而引起了广泛的学术兴趣。然而，尽管之前通过各种方法对SNN的学习梯度和模型结构进行了优化，但在性能方面SNN仍然在一定程度上落后于ANN。最近提出的多阈值模型为进一步增强SNN的学习能力提供了更多可能性。在本文中，我们从数学的角度严格分析了多阈值模型、原始脉冲模型和量化ANN之间的关系，然后提出了一种新的LM-HT模型，这是一个等距多层次模型，可以在时间维度上动态调节全局输入电流和膜电位泄漏。此外，我们指出基于LM-HT模型的直接训练算法可以无缝地连接两个阶段的学习。

    Compared to traditional Artificial Neural Network (ANN), Spiking Neural Network (SNN) has garnered widespread academic interest for its intrinsic ability to transmit information in a more biological-inspired and energy-efficient manner. However, despite previous efforts to optimize the learning gradients and model structure of SNNs through various methods, SNNs still lag behind ANNs in terms of performance to some extent. The recently proposed multi-threshold model provides more possibilities for further enhancing the learning capability of SNNs. In this paper, we rigorously analyze the relationship among the multi-threshold model, vanilla spiking model and quantized ANNs from a mathematical perspective, then propose a novel LM-HT model, which is an equidistant multi-hierarchical model that can dynamically regulate the global input current and membrane potential leakage on the time dimension. In addition, we note that the direct training algorithm based on the LM-HT model can seamlessl
    
[^11]: 融合潜变扩散模型的视频编辑：多源潜变扩散模型

    Fuse Your Latents: Video Editing with Multi-source Latent Diffusion Models. (arXiv:2310.16400v1 [cs.CV])

    [http://arxiv.org/abs/2310.16400](http://arxiv.org/abs/2310.16400)

    本论文提出了一种名为FLDM的无需训练的框架，通过融合图像 Latent Diffusion Model（LDM）和视频 LDM，在视频编辑过程中实现了文本引导的视频编辑。这一方法既保持了视频的时间一致性，又利用了图像 LDM 的高保真度，并且具有灵活性与可替换性。

    

    潜变扩散模型（LDM）以其在图像和视频合成方面的强大能力而闻名。然而，视频编辑方法存在着预训练数据不足或视频逐帧重新训练成本高的问题。为了解决这个问题，我们提出了FLDM（融合潜变扩散模型），这是一个无需训练的框架，通过在视频LDM中应用现成的图像编辑方法来实现基于文本的视频编辑。具体而言，FLDM在去噪过程中融合了图像LDM和视频LDM的潜变。这样，可以保持视频LDM的时间一致性，同时也可以利用图像LDM的高保真度。同时，由于图像LDM和视频LDM都可以替换，所以FLDM具有很高的灵活性，可以利用高级图像编辑方法，如InstructPix2Pix和ControlNet。据我们所知，FLDM是第一种将现成的图像编辑方法应用于视频LDM进行视频编辑的方法。进行了广泛的定量和定性实验。

    Latent Diffusion Models (LDMs) are renowned for their powerful capabilities in image and video synthesis. Yet, video editing methods suffer from insufficient pre-training data or video-by-video re-training cost. In addressing this gap, we propose FLDM (Fused Latent Diffusion Model), a training-free framework to achieve text-guided video editing by applying off-the-shelf image editing methods in video LDMs. Specifically, FLDM fuses latents from an image LDM and an video LDM during the denoising process. In this way, temporal consistency can be kept with video LDM while high-fidelity from the image LDM can also be exploited. Meanwhile, FLDM possesses high flexibility since both image LDM and video LDM can be replaced so advanced image editing methods such as InstructPix2Pix and ControlNet can be exploited. To the best of our knowledge, FLDM is the first method to adapt off-the-shelf image editing methods into video LDMs for video editing. Extensive quantitative and qualitative experiment
    
[^12]: 在视觉Transformer中学习成长人工海马，实现弹性终身学习

    Learning to Grow Artificial Hippocampi in Vision Transformers for Resilient Lifelong Learning. (arXiv:2303.08250v1 [cs.CV])

    [http://arxiv.org/abs/2303.08250](http://arxiv.org/abs/2303.08250)

    本文提出了一种在Vision Transformer中学习成长人工海马的方法，以实现弹性终身学习。通过神经架构搜索进行维护，选取多头自注意力块中的最终线性投影层进行ArtiHippo的实现和成长。

    

    终身学习需要拥有人类智能的韧性，即不存在灾难性遗忘，这种韧性与大脑中复杂的记忆机制，尤其是海马维护的长期记忆（LM）紧密相关。Transformer已经成为人工智能“大脑”的对应体，但LM组件在终身学习中尚未充分探索。本文提出了一种在Vision Transformer中学习成长人工海马（ArtiHippo）以实现弹性终身学习的方法。通过全面消融实验，选定多头自注意力（MHSA）块中的最终线性投影层来实现和成长ArtiHippo。ArtiHippo由专家混合（MoEs）表示。每个专家组件是线性投影层的现场变体，通过神经架构搜索（NAS）进行维护，搜索空间由四个基本成长操作（跳过、重用、适应和新）定义。

    Lifelong learning without catastrophic forgetting (i.e., resiliency) possessed by human intelligence is entangled with sophisticated memory mechanisms in the brain, especially the long-term memory (LM) maintained by Hippocampi. To a certain extent, Transformers have emerged as the counterpart ``Brain" of Artificial Intelligence (AI), and yet leave the LM component under-explored for lifelong learning settings. This paper presents a method of learning to grow Artificial Hippocampi (ArtiHippo) in Vision Transformers (ViTs) for resilient lifelong learning. With a comprehensive ablation study, the final linear projection layer in the multi-head self-attention (MHSA) block is selected in realizing and growing ArtiHippo. ArtiHippo is represented by a mixture of experts (MoEs). Each expert component is an on-site variant of the linear projection layer, maintained via neural architecture search (NAS) with the search space defined by four basic growing operations -- skip, reuse, adapt, and new 
    

