# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VCD: Knowledge Base Guided Visual Commonsense Discovery in Images](https://arxiv.org/abs/2402.17213) | 该论文提出了基于知识库的图像视觉常识发现（VCD）方法，通过定义细粒度的视觉常识类型以及构建包括超过10万张图像和1400万个对象-常识对的数据集，旨在提升计算机视觉系统的推理和决策能力。 |
| [^2] | [ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis](https://arxiv.org/abs/2402.02906) | ViewFusion 是一种用于新视角合成的最新端到端生成方法，具有无与伦比的灵活性，通过同时应用扩散去噪和像素加权掩模的方法解决了先前方法的局限性。 |
| [^3] | [Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings.](http://arxiv.org/abs/2310.17451) | 这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。 |

# 详细

[^1]: 基于知识库的图像视觉常识发现（VCD）

    VCD: Knowledge Base Guided Visual Commonsense Discovery in Images

    [https://arxiv.org/abs/2402.17213](https://arxiv.org/abs/2402.17213)

    该论文提出了基于知识库的图像视觉常识发现（VCD）方法，通过定义细粒度的视觉常识类型以及构建包括超过10万张图像和1400万个对象-常识对的数据集，旨在提升计算机视觉系统的推理和决策能力。

    

    图像中的视觉常识包含有关对象属性、关系和行为的知识。发现视觉常识可以提供对图像的更全面和丰富的理解，并增强计算机视觉系统的推理和决策能力。然而，现有的视觉常识发现研究中所定义的视觉常识是粗粒度且不完整的。在这项工作中，我们从自然语言处理中的常识知识库ConceptNet中汲取灵感，并系统地定义了各种类型的视觉常识。基于此，我们引入了一个新任务，即视觉常识发现（VCD），旨在提取图像中不同对象所包含的不同类型的细粒度常识。因此，我们从Visual Genome和ConceptNet中构建了一个名为VCDD的数据集，包括超过10万张图像和1400万个对象-常识对。

    arXiv:2402.17213v1 Announce Type: cross  Abstract: Visual commonsense contains knowledge about object properties, relationships, and behaviors in visual data. Discovering visual commonsense can provide a more comprehensive and richer understanding of images, and enhance the reasoning and decision-making capabilities of computer vision systems. However, the visual commonsense defined in existing visual commonsense discovery studies is coarse-grained and incomplete. In this work, we draw inspiration from a commonsense knowledge base ConceptNet in natural language processing, and systematically define the types of visual commonsense. Based on this, we introduce a new task, Visual Commonsense Discovery (VCD), aiming to extract fine-grained commonsense of different types contained within different objects in the image. We accordingly construct a dataset (VCDD) from Visual Genome and ConceptNet for VCD, featuring over 100,000 images and 14 million object-commonsense pairs. We furthermore pro
    
[^2]: ViewFusion: 学习可组合的扩散模型用于新视角合成

    ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis

    [https://arxiv.org/abs/2402.02906](https://arxiv.org/abs/2402.02906)

    ViewFusion 是一种用于新视角合成的最新端到端生成方法，具有无与伦比的灵活性，通过同时应用扩散去噪和像素加权掩模的方法解决了先前方法的局限性。

    

    深度学习为新视角合成这个老问题提供了丰富的新方法，从基于神经辐射场（NeRF）的方法到端到端的风格架构。每种方法都具有特定的优势，但也具有特定的适用性限制。这项工作引入了ViewFusion，这是一种具有无与伦比的灵活性的最新端到端生成方法，用于新视角合成。ViewFusion同时对场景的任意数量的输入视角应用扩散去噪步骤，然后将每个视角得到的噪声梯度与（推断得到的）像素加权掩模相结合，确保对于目标场景的每个区域，只考虑最具信息量的输入视角。我们的方法通过以下方式解决了先前方法的几个局限性：（1）可训练且能够泛化到多个场景和物体类别，（2）在训练和测试时自适应地采用可变数量的无姿态视图，（3）生成高质量的合成图像。

    Deep learning is providing a wealth of new approaches to the old problem of novel view synthesis, from Neural Radiance Field (NeRF) based approaches to end-to-end style architectures. Each approach offers specific strengths but also comes with specific limitations in their applicability. This work introduces ViewFusion, a state-of-the-art end-to-end generative approach to novel view synthesis with unparalleled flexibility. ViewFusion consists in simultaneously applying a diffusion denoising step to any number of input views of a scene, then combining the noise gradients obtained for each view with an (inferred) pixel-weighting mask, ensuring that for each region of the target scene only the most informative input views are taken into account. Our approach resolves several limitations of previous approaches by (1) being trainable and generalizing across multiple scenes and object classes, (2) adaptively taking in a variable number of pose-free views at both train and test time, (3) gene
    
[^3]: 通过理解生成：具有逻辑符号基础的神经视觉生成

    Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings. (arXiv:2310.17451v1 [cs.AI])

    [http://arxiv.org/abs/2310.17451](http://arxiv.org/abs/2310.17451)

    这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。

    

    尽管近年来神经视觉生成模型取得了很大的成功，但将其与强大的符号知识推理系统集成仍然是一个具有挑战性的任务。主要挑战有两个方面：一个是符号赋值，即将神经视觉生成器的潜在因素与知识推理系统中的有意义的符号进行绑定。另一个是规则学习，即学习新的规则，这些规则控制数据的生成过程，以增强知识推理系统。为了解决这些符号基础问题，我们提出了一种神经符号学习方法，Abductive Visual Generation (AbdGen)，用于基于诱导学习框架将逻辑编程系统与神经视觉生成模型集成起来。为了实现可靠高效的符号赋值，引入了量化诱导方法，通过语义编码本中的最近邻查找生成诱导提案。为了实现精确的规则学习，引入了对比元诱导方法。

    Despite the great success of neural visual generative models in recent years, integrating them with strong symbolic knowledge reasoning systems remains a challenging task. The main challenges are two-fold: one is symbol assignment, i.e. bonding latent factors of neural visual generators with meaningful symbols from knowledge reasoning systems. Another is rule learning, i.e. learning new rules, which govern the generative process of the data, to augment the knowledge reasoning systems. To deal with these symbol grounding problems, we propose a neural-symbolic learning approach, Abductive Visual Generation (AbdGen), for integrating logic programming systems with neural visual generative models based on the abductive learning framework. To achieve reliable and efficient symbol assignment, the quantized abduction method is introduced for generating abduction proposals by the nearest-neighbor lookups within semantic codebooks. To achieve precise rule learning, the contrastive meta-abduction
    

