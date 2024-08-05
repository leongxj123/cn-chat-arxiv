# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Optimization Framework to Enforce Multi-View Consistency for Texturing 3D Meshes Using Pre-Trained Text-to-Image Models](https://arxiv.org/abs/2403.15559) | 该论文介绍了一个四阶段的优化框架，通过MV一致的扩散过程、半定编程问题解决、非刚性对齐和MRF问题解决等步骤来实现对3D网格进行纹理贴图的多视图一致性。 |
| [^2] | [Adaptive Self-training Framework for Fine-grained Scene Graph Generation.](http://arxiv.org/abs/2401.09786) | 本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。 |
| [^3] | [SynthoGestures: A Novel Framework for Synthetic Dynamic Hand Gesture Generation for Driving Scenarios.](http://arxiv.org/abs/2309.04421) | SynthoGestures是一种使用虚幻引擎合成逼真手势的新框架，可以用于驾驶场景下的动态人机界面。该框架通过生成多种变体和模拟不同摄像机类型，提高了手势识别的准确性并节省了数据集创建的时间和精力。 |
| [^4] | [Multi-Modality Guidance Network For Missing Modality Inference.](http://arxiv.org/abs/2309.03452) | 我们提出了一种多模态辅助网络，在训练过程中利用多模态表示来训练更好的单模态模型进行推断，解决了推断过程中模态缺失的问题。实验结果表明，我们的方法在性能上显著优于传统方法。 |

# 详细

[^1]: 一个优化框架，利用预训练的文本到图像模型强制实现对3D网格进行纹理贴图的多视图一致性

    An Optimization Framework to Enforce Multi-View Consistency for Texturing 3D Meshes Using Pre-Trained Text-to-Image Models

    [https://arxiv.org/abs/2403.15559](https://arxiv.org/abs/2403.15559)

    该论文介绍了一个四阶段的优化框架，通过MV一致的扩散过程、半定编程问题解决、非刚性对齐和MRF问题解决等步骤来实现对3D网格进行纹理贴图的多视图一致性。

    

    在使用预训练的文本到图像模型对3D网格进行纹理贴图时，确保多视图一致性是一个基本问题。本文介绍了一个优化框架，通过四个阶段实现多视图一致性。具体而言，第一阶段使用MV一致的扩散过程从预定义的视点集生成2D纹理的过完备集。第二阶段通过解决半定编程问题选择相互一致且覆盖基础3D模型的视图子集。第三阶段执行非刚性对齐，使选定的视图在重叠区域对齐。第四阶段解决MRF问题以关联...

    arXiv:2403.15559v1 Announce Type: cross  Abstract: A fundamental problem in the texturing of 3D meshes using pre-trained text-to-image models is to ensure multi-view consistency. State-of-the-art approaches typically use diffusion models to aggregate multi-view inputs, where common issues are the blurriness caused by the averaging operation in the aggregation step or inconsistencies in local features. This paper introduces an optimization framework that proceeds in four stages to achieve multi-view consistency. Specifically, the first stage generates an over-complete set of 2D textures from a predefined set of viewpoints using an MV-consistent diffusion process. The second stage selects a subset of views that are mutually consistent while covering the underlying 3D model. We show how to achieve this goal by solving semi-definite programs. The third stage performs non-rigid alignment to align the selected views across overlapping regions. The fourth stage solves an MRF problem to associ
    
[^2]: 自适应自训练框架用于细粒度场景图生成

    Adaptive Self-training Framework for Fine-grained Scene Graph Generation. (arXiv:2401.09786v1 [cs.CV])

    [http://arxiv.org/abs/2401.09786](http://arxiv.org/abs/2401.09786)

    本论文提出了一种自适应自训练框架用于细粒度场景图生成，通过利用未标注的三元组缓解了场景图生成中的长尾问题。同时，引入了一种新颖的伪标签技术CATM和图结构学习器GSL来提高模型性能。

    

    场景图生成（SGG）模型在基准数据集中存在长尾谓词分布和缺失注释问题。本研究旨在通过利用未标注的三元组缓解SGG的长尾问题。为此，我们引入了一种称为自训练SGG（ST-SGG）的框架，该框架基于未标注的三元组为其分配伪标签以训练SGG模型。虽然在图像识别方面的自训练取得了显著进展，但设计适用于SGG任务的自训练框架更具挑战，因为其固有特性，如语义歧义和长尾分布的谓词类别。因此，我们提出了一种新颖的SGG伪标签技术，称为具有动量的类别自适应阈值化（CATM），它是一种独立于模型的框架，可应用于任何已有的SGG模型。此外，我们设计了一个图结构学习器（GSL），从中获益。

    Scene graph generation (SGG) models have suffered from inherent problems regarding the benchmark datasets such as the long-tailed predicate distribution and missing annotation problems. In this work, we aim to alleviate the long-tailed problem of SGG by utilizing unannotated triplets. To this end, we introduce a Self-Training framework for SGG (ST-SGG) that assigns pseudo-labels for unannotated triplets based on which the SGG models are trained. While there has been significant progress in self-training for image recognition, designing a self-training framework for the SGG task is more challenging due to its inherent nature such as the semantic ambiguity and the long-tailed distribution of predicate classes. Hence, we propose a novel pseudo-labeling technique for SGG, called Class-specific Adaptive Thresholding with Momentum (CATM), which is a model-agnostic framework that can be applied to any existing SGG models. Furthermore, we devise a graph structure learner (GSL) that is benefici
    
[^3]: SynthoGestures：一种用于驾驶场景的合成动态手势生成的新框架

    SynthoGestures: A Novel Framework for Synthetic Dynamic Hand Gesture Generation for Driving Scenarios. (arXiv:2309.04421v1 [cs.CV])

    [http://arxiv.org/abs/2309.04421](http://arxiv.org/abs/2309.04421)

    SynthoGestures是一种使用虚幻引擎合成逼真手势的新框架，可以用于驾驶场景下的动态人机界面。该框架通过生成多种变体和模拟不同摄像机类型，提高了手势识别的准确性并节省了数据集创建的时间和精力。

    

    在汽车领域中，为动态人机界面创建多样化和全面的手势数据集可能具有挑战性且耗时。为了克服这一挑战，我们提出使用虚拟3D模型生成合成手势数据集。我们的框架利用虚幻引擎合成逼真的手势，提供定制选项并降低过拟合风险。生成多种变体，包括手势速度、性能和手形，以提高泛化能力。此外，我们模拟不同的摄像机位置和类型，如RGB、红外和深度摄像机，而无需额外的时间和费用获取这些摄像机。实验结果表明，我们的提议框架SynthoGestures提高了手势识别准确率，可以替代或增强真手数据集。通过节省数据集创建的时间和精力，我们的工具促进了研究的进展。

    Creating a diverse and comprehensive dataset of hand gestures for dynamic human-machine interfaces in the automotive domain can be challenging and time-consuming. To overcome this challenge, we propose using synthetic gesture datasets generated by virtual 3D models. Our framework utilizes Unreal Engine to synthesize realistic hand gestures, offering customization options and reducing the risk of overfitting. Multiple variants, including gesture speed, performance, and hand shape, are generated to improve generalizability. In addition, we simulate different camera locations and types, such as RGB, infrared, and depth cameras, without incurring additional time and cost to obtain these cameras. Experimental results demonstrate that our proposed framework, SynthoGestures\footnote{\url{https://github.com/amrgomaaelhady/SynthoGestures}}, improves gesture recognition accuracy and can replace or augment real-hand datasets. By saving time and effort in the creation of the data set, our tool acc
    
[^4]: 多模态辅助网络用于推断缺失模态

    Multi-Modality Guidance Network For Missing Modality Inference. (arXiv:2309.03452v1 [cs.CV])

    [http://arxiv.org/abs/2309.03452](http://arxiv.org/abs/2309.03452)

    我们提出了一种多模态辅助网络，在训练过程中利用多模态表示来训练更好的单模态模型进行推断，解决了推断过程中模态缺失的问题。实验结果表明，我们的方法在性能上显著优于传统方法。

    

    多模态模型在最近几年取得了显著的成功。标准多模态方法通常假设在训练阶段和推断阶段模态保持不变。然而在实践中，许多场景无法满足这样的假设，推断过程中会出现缺失模态，从而限制了多模态模型的应用范围。现有的方法通过重建缺失模态来缓解这个问题，但这增加了不必要的计算成本，尤其对于大型部署系统而言可能是至关重要的。为了从两个方面解决这个问题，我们提出了一种新颖的辅助网络，在训练过程中促进知识共享，利用多模态表示来训练更好的单模态模型进行推断。在暴力检测的现实生活实验中，我们提出的框架训练的单模态模型在性能上显著优于传统训练的模型，同时保持相同的推断能力。

    Multimodal models have gained significant success in recent years. Standard multimodal approaches often assume unchanged modalities from training stage to inference stage. In practice, however, many scenarios fail to satisfy such assumptions with missing modalities during inference, leading to limitations on where multimodal models can be applied. While existing methods mitigate the problem through reconstructing the missing modalities, it increases unnecessary computational cost, which could be just as critical, especially for large, deployed systems. To solve the problem from both sides, we propose a novel guidance network that promotes knowledge sharing during training, taking advantage of the multimodal representations to train better single-modality models for inference. Real-life experiment in violence detection shows that our proposed framework trains single-modality models that significantly outperform its traditionally trained counterparts while maintaining the same inference 
    

