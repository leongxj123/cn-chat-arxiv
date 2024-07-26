# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ReMamber: Referring Image Segmentation with Mamba Twister](https://arxiv.org/abs/2403.17839) | 提出了ReMamber，一种整合了Mamba和多模态Mamba Twister块的新型RIS架构，通过其独特的通道和空间扭曲机制实现图像-文本交互，取得了三个基准测试的最新技术成果 |
| [^2] | [DiCoM -- Diverse Concept Modeling towards Enhancing Generalizability in Chest X-Ray Studies](https://arxiv.org/abs/2402.15534) | DiCoM是一种新颖的自监督训练范式，通过学习多元概念，有效表示胸部X射线数据，以应对医学成像预训练中与自然图像不同的挑战。 |
| [^3] | [3D Diffuser Actor: Policy Diffusion with 3D Scene Representations](https://arxiv.org/abs/2402.10885) | 通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。 |
| [^4] | [Self-supervised learning of video representations from a child's perspective](https://arxiv.org/abs/2402.00300) | 本研究从儿童的视角进行自监督学习，通过长时间的头戴式摄像记录训练视频模型，结果表明这些模型在促进从少量样本中学习行动概念方面非常有效。 |
| [^5] | [Generalized Continual Category Discovery.](http://arxiv.org/abs/2308.12112) | 本研究提出了一种广义持续类别发现（GCCD）的框架，用于在现实生活场景中同时处理新的和已知的类别，并且利用持续的无监督学习方法来发现它们。通过实验证明现有方法无法处理后续任务中的无标记样本。 |

# 详细

[^1]: ReMamber：使用Mamba Twister实现引用图像分割

    ReMamber: Referring Image Segmentation with Mamba Twister

    [https://arxiv.org/abs/2403.17839](https://arxiv.org/abs/2403.17839)

    提出了ReMamber，一种整合了Mamba和多模态Mamba Twister块的新型RIS架构，通过其独特的通道和空间扭曲机制实现图像-文本交互，取得了三个基准测试的最新技术成果

    

    引用图像分割（RIS）利用变换器在解释复杂的视觉-语言任务方面取得了巨大成功。然而，二次计算成本使其在捕捉远程视觉-语言依赖性方面消耗资源。幸运的是，Mamba通过高效的线性复杂度在处理方面解决了这个问题。然而，将Mamba直接应用于多模态交互会面临挑战，主要原因是因为通道交互不足，无法有效融合多模态数据。在本文中，我们提出了ReMamber，这是一种整合了Mamba和多模态Mamba Twister块强大功能的新型RIS架构。Mamba Twister通过其独特的通道和空间扭曲机制明确建模图像-文本交互，并通过其独特的通道和空间扭曲机制融合文本和视觉特征。我们在三个具有挑战性的基准测试中取得了最新技术成果。此外，我们对ReMamber进行了彻底分析，并讨论其他...

    arXiv:2403.17839v1 Announce Type: cross  Abstract: Referring Image Segmentation (RIS) leveraging transformers has achieved great success on the interpretation of complex visual-language tasks. However, the quadratic computation cost makes it resource-consuming in capturing long-range visual-language dependencies. Fortunately, Mamba addresses this with efficient linear complexity in processing. However, directly applying Mamba to multi-modal interactions presents challenges, primarily due to inadequate channel interactions for the effective fusion of multi-modal data. In this paper, we propose ReMamber, a novel RIS architecture that integrates the power of Mamba with a multi-modal Mamba Twister block. The Mamba Twister explicitly models image-text interaction, and fuses textual and visual features through its unique channel and spatial twisting mechanism. We achieve the state-of-the-art on three challenging benchmarks. Moreover, we conduct thorough analyses of ReMamber and discuss other
    
[^2]: DiCoM -- 多元概念建模以增强胸部X射线研究的普适性

    DiCoM -- Diverse Concept Modeling towards Enhancing Generalizability in Chest X-Ray Studies

    [https://arxiv.org/abs/2402.15534](https://arxiv.org/abs/2402.15534)

    DiCoM是一种新颖的自监督训练范式，通过学习多元概念，有效表示胸部X射线数据，以应对医学成像预训练中与自然图像不同的挑战。

    

    胸部X线（CXR）是一种广泛应用的临床成像模态，在各种肺部和心脏相关疾病的诊断和预后中起着关键作用。传统的依赖放射学读片和监督学习的自动化临床诊断工具设计策略需要高质量注释训练数据，为了解决这一挑战，自监督预训练已被证明在许多下游视觉任务中胜过监督预训练，代表了该领域的重大突破。然而，医学成像预训练与自然图像（例如ImageNet）的预训练在很大程度上不同，因为临床图像具有独特属性。在这种背景下，我们介绍了多元概念建模（DiCoM），这是一种新颖的自监督训练范式，利用了学生教师框架来学习多元概念，从而有效表示CXR数据。

    arXiv:2402.15534v1 Announce Type: cross  Abstract: Chest X-Ray (CXR) is a widely used clinical imaging modality and has a pivotal role in the diagnosis and prognosis of various lung and heart related conditions. Conventional automated clinical diagnostic tool design strategies relying on radiology reads and supervised learning, entail the cumbersome requirement of high quality annotated training data. To address this challenge, self-supervised pre-training has proven to outperform supervised pre-training in numerous downstream vision tasks, representing a significant breakthrough in the field. However, medical imaging pre-training significantly differs from pre-training with natural images (e.g., ImageNet) due to unique attributes of clinical images. In this context, we introduce Diverse Concept Modeling (DiCoM), a novel self-supervised training paradigm that leverages a student teacher framework for learning diverse concepts and hence effective representation of the CXR data. Hence, e
    
[^3]: 基于3D场景表示的3D扩散器Actor：通过策略扩散进行机器人操作

    3D Diffuser Actor: Policy Diffusion with 3D Scene Representations

    [https://arxiv.org/abs/2402.10885](https://arxiv.org/abs/2402.10885)

    通过策略扩散和3D场景表示相结合，提出了3D Diffuser Actor，一个神经策略架构，可以根据语言指令构建3D视觉场景表示，并对机器人末端执行器的3D旋转和平移进行迭代去噪。

    

    我们将扩散策略和3D场景表示相结合，用于机器人操作。扩散策略通过条件扩散模型学习基于机器人和环境状态的动作分布。最近，它们已经表现出优于确定性和其他基于状态的动作分布学习方法。3D机器人策略使用从单个或多个摄像头视角获取的感应深度聚合的3D场景特征表示。它们已经证明比其2D对应物在摄像机视角上具有更好的泛化能力。我们统一了这两条线路的工作，并提出了3D扩散器Actor，这是一个神经策略架构，它在给定语言指令的情况下，构建视觉场景的3D表示，并在其上进行条件迭代去噪机器人末端执行器的3D旋转和平移。在每个去噪迭代中，我们的模型将末端执行器姿态估计表示为3D场景令牌，并预测t

    arXiv:2402.10885v1 Announce Type: cross  Abstract: We marry diffusion policies and 3D scene representations for robot manipulation. Diffusion policies learn the action distribution conditioned on the robot and environment state using conditional diffusion models. They have recently shown to outperform both deterministic and alternative state-conditioned action distribution learning methods. 3D robot policies use 3D scene feature representations aggregated from a single or multiple camera views using sensed depth. They have shown to generalize better than their 2D counterparts across camera viewpoints. We unify these two lines of work and present 3D Diffuser Actor, a neural policy architecture that, given a language instruction, builds a 3D representation of the visual scene and conditions on it to iteratively denoise 3D rotations and translations for the robot's end-effector. At each denoising iteration, our model represents end-effector pose estimates as 3D scene tokens and predicts t
    
[^4]: 从儿童视角进行自监督学习的视频表示

    Self-supervised learning of video representations from a child's perspective

    [https://arxiv.org/abs/2402.00300](https://arxiv.org/abs/2402.00300)

    本研究从儿童的视角进行自监督学习，通过长时间的头戴式摄像记录训练视频模型，结果表明这些模型在促进从少量样本中学习行动概念方面非常有效。

    

    儿童通过几年的自我视觉经验学习到了强大的世界内部模型。这些内部模型能否通过儿童的视觉体验和通用的自监督学习算法来学习，还是需要强大的归纳偏差？最近，在收集大规模、纵向的发展现实视频数据集以及通用的自监督学习算法的进展使我们能够开始探讨这个本质与养育之间的问题。然而，现有的工作通常关注基于图像的自监督学习算法和可以从静态图像中学习的视觉能力（例如目标识别），从而忽略了世界的时间性质。为了弥合这一差距，我们在一个儿童早期发展阶段（6-31个月）从儿童的头戴式摄像记录中训练自监督视频模型。所得到的模型在促进从少量样本中学习行动概念方面非常有效。

    Children learn powerful internal models of the world around them from a few years of egocentric visual experience. Can such internal models be learned from a child's visual experience with highly generic learning algorithms or do they require strong inductive biases? Recent advances in collecting large-scale, longitudinal, developmentally realistic video datasets and generic self-supervised learning (SSL) algorithms are allowing us to begin to tackle this nature vs. nurture question. However, existing work typically focuses on image-based SSL algorithms and visual capabilities that can be learned from static images (e.g. object recognition), thus ignoring temporal aspects of the world. To close this gap, here we train self-supervised video models on longitudinal, egocentric headcam recordings collected from a child over a two year period in their early development (6-31 months). The resulting models are highly effective at facilitating the learning of action concepts from a small numbe
    
[^5]: 广义持续类别发现

    Generalized Continual Category Discovery. (arXiv:2308.12112v1 [cs.LG])

    [http://arxiv.org/abs/2308.12112](http://arxiv.org/abs/2308.12112)

    本研究提出了一种广义持续类别发现（GCCD）的框架，用于在现实生活场景中同时处理新的和已知的类别，并且利用持续的无监督学习方法来发现它们。通过实验证明现有方法无法处理后续任务中的无标记样本。

    

    大多数持续学习（CL）方法推动着监督学习设置的极限，其中一个智能体期望学习新的标记任务而不会忘记先前的知识。然而，这些设置与现实生活场景不太吻合，其中学习智能体可以访问大量的无标记数据，包括全新（完全无标记）类别和已知类别的示例。受到广义类别发现（GCD）的启发，我们引入了一个新的框架来放松这个假设。确切地说，在任何任务中，我们允许存在新的和已知的类别，并且必须使用持续版本的无监督学习方法来发现它们。我们称这种设置为广义持续类别发现（GCCD）。它统一了CL和GCD，弥合了合成基准和现实生活场景之间的差距。通过一系列实验，我们发现现有的方法无法从后续任务中积累知识，其中包含无标记样本。

    Most of Continual Learning (CL) methods push the limit of supervised learning settings, where an agent is expected to learn new labeled tasks and not forget previous knowledge. However, these settings are not well aligned with real-life scenarios, where a learning agent has access to a vast amount of unlabeled data encompassing both novel (entirely unlabeled) classes and examples from known classes. Drawing inspiration from Generalized Category Discovery (GCD), we introduce a novel framework that relaxes this assumption. Precisely, in any task, we allow for the existence of novel and known classes, and one must use continual version of unsupervised learning methods to discover them. We call this setting Generalized Continual Category Discovery (GCCD). It unifies CL and GCD, bridging the gap between synthetic benchmarks and real-life scenarios. With a series of experiments, we present that existing methods fail to accumulate knowledge from subsequent tasks in which unlabeled samples of 
    

