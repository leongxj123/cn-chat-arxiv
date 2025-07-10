# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^2] | [Geometric Constraints in Deep Learning Frameworks: A Survey](https://arxiv.org/abs/2403.12431) | 本调查研究了几何约束和深度学习框架之间的重合部分，比较了深度估计等问题中集成在深度学习框架中的几何强制约束。 |
| [^3] | [From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution.](http://arxiv.org/abs/2401.14661) | 基于超分辨率和经过调整的轻量级YOLOv5架构，我们提出了一种创新的方法来解决航空影像中小而密集物体检测的挑战。我们的超分辨率YOLOv5模型采用Transformer编码器块，能够捕捉全局背景和上下文信息，从而在高密度、遮挡条件下提高检测结果。这种轻量级模型不仅准确性更高，而且资源利用效率高，非常适合实时应用。 |
| [^4] | [Rethinking Class-incremental Learning in the Era of Large Pre-trained Models via Test-Time Adaptation.](http://arxiv.org/abs/2310.11482) | 本研究提出了一种名为“增量学习的测试时适应”的方法，通过在测试实例上进行微调，避免了在每个新任务上进行训练，从而在增量学习中实现了预训练模型的稳定性和可塑性的平衡。 |

# 详细

[^1]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^2]: 深度学习框架中的几何约束：一项调查

    Geometric Constraints in Deep Learning Frameworks: A Survey

    [https://arxiv.org/abs/2403.12431](https://arxiv.org/abs/2403.12431)

    本调查研究了几何约束和深度学习框架之间的重合部分，比较了深度估计等问题中集成在深度学习框架中的几何强制约束。

    

    Stereophotogrammetry是一种新兴的场景理解技术。其起源可以追溯到至少19世纪，当时人们开始研究使用照片来测量世界的物理属性。自那时以来，已经探索了成千上万种方法。经典几何技术的Shape from Stereo建立在使用几何来定义场景和摄像机几何的约束，然后解决非线性方程组。更近期的工作采用了完全不同的方法，使用端到端的深度学习而没有明确建模几何。在这项调查中，我们探讨了基于几何和基于深度学习框架的重叠部分。我们比较和对比了集成到深度学习框架中用于深度估计或其他密切相关问题的几何强制约束。我们提出了一种新的分类法，用于描述现代深度学习中使用的普遍几何约束。

    arXiv:2403.12431v1 Announce Type: cross  Abstract: Stereophotogrammetry is an emerging technique of scene understanding. Its origins go back to at least the 1800s when people first started to investigate using photographs to measure the physical properties of the world. Since then, thousands of approaches have been explored. The classic geometric techniques of Shape from Stereo is built on using geometry to define constraints on scene and camera geometry and then solving the non-linear systems of equations. More recent work has taken an entirely different approach, using end-to-end deep learning without any attempt to explicitly model the geometry. In this survey, we explore the overlap for geometric-based and deep learning-based frameworks. We compare and contrast geometry enforcing constraints integrated into a deep learning framework for depth estimation or other closely related problems. We present a new taxonomy for prevalent geometry enforcing constraints used in modern deep lear
    
[^3]: 从模糊到明亮的检测：基于YOLOv5的超分辨率航空物体检测

    From Blurry to Brilliant Detection: YOLOv5-Based Aerial Object Detection with Super Resolution. (arXiv:2401.14661v1 [cs.CV])

    [http://arxiv.org/abs/2401.14661](http://arxiv.org/abs/2401.14661)

    基于超分辨率和经过调整的轻量级YOLOv5架构，我们提出了一种创新的方法来解决航空影像中小而密集物体检测的挑战。我们的超分辨率YOLOv5模型采用Transformer编码器块，能够捕捉全局背景和上下文信息，从而在高密度、遮挡条件下提高检测结果。这种轻量级模型不仅准确性更高，而且资源利用效率高，非常适合实时应用。

    

    随着无人机和卫星技术的广泛应用，对航空影像中准确物体检测的需求大大增加。传统的物体检测模型在偏向大物体的数据集上训练，对于航空场景中普遍存在的小而密集的物体难以发挥最佳性能。为了解决这个挑战，我们提出了一种创新的方法，结合了超分辨率和经过调整的轻量级YOLOv5架构。我们使用多种数据集进行评估，包括VisDrone-2023、SeaDroneSee、VEDAI和NWPU VHR-10，以验证我们模型的性能。我们的超分辨率YOLOv5架构采用Transformer编码器块，使模型能够捕捉到全局背景和上下文信息，从而提高检测结果，特别是在高密度、遮挡条件下。这种轻量级模型不仅提供了更高的准确性，还确保了资源的有效利用，非常适合实时应用。我们的实验表明，我们的模型在航空物体检测任务中表现出色，特别是在复杂场景中。

    The demand for accurate object detection in aerial imagery has surged with the widespread use of drones and satellite technology. Traditional object detection models, trained on datasets biased towards large objects, struggle to perform optimally in aerial scenarios where small, densely clustered objects are prevalent. To address this challenge, we present an innovative approach that combines super-resolution and an adapted lightweight YOLOv5 architecture. We employ a range of datasets, including VisDrone-2023, SeaDroneSee, VEDAI, and NWPU VHR-10, to evaluate our model's performance. Our Super Resolved YOLOv5 architecture features Transformer encoder blocks, allowing the model to capture global context and context information, leading to improved detection results, especially in high-density, occluded conditions. This lightweight model not only delivers improved accuracy but also ensures efficient resource utilization, making it well-suited for real-time applications. Our experimental 
    
[^4]: 在大型预训练模型时代重新思考增量学习的测试时适应方法

    Rethinking Class-incremental Learning in the Era of Large Pre-trained Models via Test-Time Adaptation. (arXiv:2310.11482v1 [cs.CV])

    [http://arxiv.org/abs/2310.11482](http://arxiv.org/abs/2310.11482)

    本研究提出了一种名为“增量学习的测试时适应”的方法，通过在测试实例上进行微调，避免了在每个新任务上进行训练，从而在增量学习中实现了预训练模型的稳定性和可塑性的平衡。

    

    增量学习是一个具有挑战性的任务，涉及持续学习将类别划分到新任务中，同时不会遗忘先前学到的信息。大型预训练模型的出现加快了增量学习的进展，因为高度可传输的预训练模型表示使得在调整一小组参数时，与从头开始训练的传统增量学习方法相比，可以获得最先进的性能。然而，对每个任务进行反复微调会破坏预训练模型的丰富表示，并导致遗忘之前的任务。为了在增量学习中在预训练模型的稳定性和可塑性之间取得平衡，我们提出了一种新颖的方法，即通过直接在测试实例上进行测试时适应。具体而言，我们提出了“增量学习的测试时适应”（TTACIL），它首先在每个测试实例上对预训练模型的层归一化参数进行微调。

    Class-incremental learning (CIL) is a challenging task that involves continually learning to categorize classes into new tasks without forgetting previously learned information. The advent of the large pre-trained models (PTMs) has fast-tracked the progress in CIL due to the highly transferable PTM representations, where tuning a small set of parameters results in state-of-the-art performance when compared with the traditional CIL methods that are trained from scratch. However, repeated fine-tuning on each task destroys the rich representations of the PTMs and further leads to forgetting previous tasks. To strike a balance between the stability and plasticity of PTMs for CIL, we propose a novel perspective of eliminating training on every new task and instead performing test-time adaptation (TTA) directly on the test instances. Concretely, we propose "Test-Time Adaptation for Class-Incremental Learning" (TTACIL) that first fine-tunes Layer Norm parameters of the PTM on each test instan
    

