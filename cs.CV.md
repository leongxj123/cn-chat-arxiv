# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NIGHT -- Non-Line-of-Sight Imaging from Indirect Time of Flight Data](https://arxiv.org/abs/2403.19376) | 本文首次使用来自即插即用的间接飞行时间传感器的数据，引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子，从而实现了获取隐藏场景深度信息的可行性。 |
| [^2] | [On the Utility of 3D Hand Poses for Action Recognition](https://arxiv.org/abs/2403.09805) | 提出了一种名为HandFormer的新型多模态Transformer模型，结合了高时间分辨率的3D手部姿势和稀疏采样的RGB帧，用于有效建模手部和物体之间的相互作用，取得了很高的准确性。 |
| [^3] | [VIRUS-NeRF -- Vision, InfraRed and UltraSonic based Neural Radiance Fields](https://arxiv.org/abs/2403.09477) | VIRUS-NeRF是基于视觉、红外和超声波的神经辐射场，通过整合超声波和红外传感器的深度测量数据，实现了在自主移动机器人中达到与LiDAR点云相媲美的映射性能。 |
| [^4] | [Distilling the Knowledge in Data Pruning](https://arxiv.org/abs/2403.07854) | 在数据剪枝中引入知识蒸馏方法，通过与预先训练的教师网络软预测相结合，实现了在各种数据集、剪枝方法和所有剪枝分数上的显著提升。 |
| [^5] | [CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering](https://arxiv.org/abs/2403.00816) | 该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。 |
| [^6] | [Disentangled Representation Learning with Transmitted Information Bottleneck.](http://arxiv.org/abs/2311.01686) | 本研究提出了一种通过引入传输信息瓶颈来实现解缠表示学习的方法。该方法可以在压缩表示信息和保留重要信息之间维持平衡，从而提高模型的稳健性和泛化能力。通过使用贝叶斯网络和变分推断，我们得到了可计算估计的DisTIB。 |
| [^7] | [AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models.](http://arxiv.org/abs/2309.16414) | 本研究提出了一种名为AutoCLIP的方法，用于自动调谐视觉语言模型的零样本分类器。AutoCLIP通过为每个提示模板分配图像特定的权重，从而改进了从编码类别描述符推导零样本分类器的方式。 |
| [^8] | [Self-Supervised Scalable Deep Compressed Sensing.](http://arxiv.org/abs/2308.13777) | 本文提出了一种自监督的可扩展深度压缩感知方法，不需要标记的测量-地面真实数据，并且可以处理任意的采样比率和矩阵。该方法包括一个双域损失和四个恢复阶段，通过最大化数据/信息利用率来提高准确性。 |
| [^9] | [Patch-wise Auto-Encoder for Visual Anomaly Detection.](http://arxiv.org/abs/2308.00429) | 本论文提出了一种新颖的补丁化自编码器（Patch AE）框架来增强自编码器对异常的重构能力，并在Mvtec AD基准测试中取得了最先进的表现，具有在实际工业应用场景中的潜力。 |

# 详细

[^1]: NIGHT -- 间接飞行时间数据的非视距成像

    NIGHT -- Non-Line-of-Sight Imaging from Indirect Time of Flight Data

    [https://arxiv.org/abs/2403.19376](https://arxiv.org/abs/2403.19376)

    本文首次使用来自即插即用的间接飞行时间传感器的数据，引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子，从而实现了获取隐藏场景深度信息的可行性。

    

    从非视角相机外部获取物体是一个非常引人注目但也极具挑战性的研究课题。最近的工作表明，利用定制的直接飞行时间传感器产生的瞬时成像数据，这个想法是可行的。在本文中，我们首次使用来自即插即用的间接飞行时间传感器的数据来解决这个问题，而不需要任何额外的硬件要求。我们引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子。这种建模使得任务更容易处理，也有助于构建带有注释的训练数据。从获得的数据中，可以恢复隐藏场景的深度信息。我们还提供了一个首创的合成数据集用于这个任务，并展示了所提出的想法的可行性。

    arXiv:2403.19376v1 Announce Type: cross  Abstract: The acquisition of objects outside the Line-of-Sight of cameras is a very intriguing but also extremely challenging research topic. Recent works showed the feasibility of this idea exploiting transient imaging data produced by custom direct Time of Flight sensors. In this paper, for the first time, we tackle this problem using only data from an off-the-shelf indirect Time of Flight sensor without any further hardware requirement. We introduced a Deep Learning model able to re-frame the surfaces where light bounces happen as a virtual mirror. This modeling makes the task easier to handle and also facilitates the construction of annotated training data. From the obtained data it is possible to retrieve the depth information of the hidden scene. We also provide a first-in-its-kind synthetic dataset for the task and demonstrate the feasibility of the proposed idea over it.
    
[^2]: 关于3D手部姿势在动作识别中的作用

    On the Utility of 3D Hand Poses for Action Recognition

    [https://arxiv.org/abs/2403.09805](https://arxiv.org/abs/2403.09805)

    提出了一种名为HandFormer的新型多模态Transformer模型，结合了高时间分辨率的3D手部姿势和稀疏采样的RGB帧，用于有效建模手部和物体之间的相互作用，取得了很高的准确性。

    

    3D手部姿势是一种未充分探索的动作识别模态。姿势既紧凑又信息丰富，并且可以极大地受益于计算预算有限的应用。然而，单独的姿势不能完全理解人类与之交互的物体和环境。为了有效建模手部物体相互作用，我们提出了HandFormer，一种新颖的多模态Transformer。HandFormer结合了高时间分辨率的3D手部姿势，用于精细运动建模，并使用稀疏采样的RGB帧来编码场景语义。观察手部姿势的独特特征，我们对手部建模进行了时间分解，并通过其短期轨迹表示每个关节点。这种被分解的姿势表示与稀疏的RGB采样相结合，效率非常高，并且达到了很高的准确性。仅有手部姿势的单模HandFormer在5倍更少的FLO下胜过现有基于骨架的方法。

    arXiv:2403.09805v1 Announce Type: cross  Abstract: 3D hand poses are an under-explored modality for action recognition. Poses are compact yet informative and can greatly benefit applications with limited compute budgets. However, poses alone offer an incomplete understanding of actions, as they cannot fully capture objects and environments with which humans interact. To efficiently model hand-object interactions, we propose HandFormer, a novel multimodal transformer. HandFormer combines 3D hand poses at a high temporal resolution for fine-grained motion modeling with sparsely sampled RGB frames for encoding scene semantics. Observing the unique characteristics of hand poses, we temporally factorize hand modeling and represent each joint by its short-term trajectories. This factorized pose representation combined with sparse RGB samples is remarkably efficient and achieves high accuracy. Unimodal HandFormer with only hand poses outperforms existing skeleton-based methods at 5x fewer FLO
    
[^3]: 基于视觉、红外和超声波的神经辐射场——VIRUS-NeRF

    VIRUS-NeRF -- Vision, InfraRed and UltraSonic based Neural Radiance Fields

    [https://arxiv.org/abs/2403.09477](https://arxiv.org/abs/2403.09477)

    VIRUS-NeRF是基于视觉、红外和超声波的神经辐射场，通过整合超声波和红外传感器的深度测量数据，实现了在自主移动机器人中达到与LiDAR点云相媲美的映射性能。

    

    自主移动机器人在现代工厂和仓库操作中起着越来越重要的作用。障碍物检测、回避和路径规划是关键的安全相关任务，通常使用昂贵的LiDAR传感器和深度摄像头来解决。我们提出使用成本效益的低分辨率测距传感器，如超声波和红外时间飞行传感器，通过开发基于视觉、红外和超声波的神经辐射场(VIRUS-NeRF)来解决这一问题。VIRUS-NeRF构建在瞬时神经图形基元与多分辨率哈希编码(Instant-NGP)的基础上，融合了超声波和红外传感器的深度测量数据，并利用它们来更新用于光线跟踪的占据网格。在2D实验评估中，VIRUS-NeRF实现了与LiDAR点云相媲美的映射性能，尤其在小型环境中，其准确性与LiDAR测量相符。

    arXiv:2403.09477v1 Announce Type: cross  Abstract: Autonomous mobile robots are an increasingly integral part of modern factory and warehouse operations. Obstacle detection, avoidance and path planning are critical safety-relevant tasks, which are often solved using expensive LiDAR sensors and depth cameras. We propose to use cost-effective low-resolution ranging sensors, such as ultrasonic and infrared time-of-flight sensors by developing VIRUS-NeRF - Vision, InfraRed, and UltraSonic based Neural Radiance Fields. Building upon Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (Instant-NGP), VIRUS-NeRF incorporates depth measurements from ultrasonic and infrared sensors and utilizes them to update the occupancy grid used for ray marching. Experimental evaluation in 2D demonstrates that VIRUS-NeRF achieves comparable mapping performance to LiDAR point clouds regarding coverage. Notably, in small environments, its accuracy aligns with that of LiDAR measurements, whi
    
[^4]: 在数据剪枝中蒸馏知识

    Distilling the Knowledge in Data Pruning

    [https://arxiv.org/abs/2403.07854](https://arxiv.org/abs/2403.07854)

    在数据剪枝中引入知识蒸馏方法，通过与预先训练的教师网络软预测相结合，实现了在各种数据集、剪枝方法和所有剪枝分数上的显著提升。

    

    随着训练神经网络使用的数据集规模不断增加，数据剪枝成为了一个有吸引力的研究领域。然而，大多数当前的数据剪枝算法在保持准确性方面受到限制，特别是在高度剪枝的情况下与使用完整数据训练的模型相比。本文探讨了在训练基于剪枝子集的模型时，结合知识蒸馏（KD）的应用。也就是说，我们不仅依赖于地面真实标签，还使用了已在完整数据上预先训练的老师网络的软预测。通过将知识蒸馏整合到训练中，我们在各种数据集、剪枝方法和所有剪枝分数上都展示了显著的改进。我们首先建立了采用自蒸馏来改善在剪枝数据上的训练的理论动机。然后，我们在实证上进行了引人注目且高度实用的观察：使用知识蒸馏，简单的随机剪枝也会取得显着改进。

    arXiv:2403.07854v1 Announce Type: cross  Abstract: With the increasing size of datasets used for training neural networks, data pruning becomes an attractive field of research. However, most current data pruning algorithms are limited in their ability to preserve accuracy compared to models trained on the full data, especially in high pruning regimes. In this paper we explore the application of data pruning while incorporating knowledge distillation (KD) when training on a pruned subset. That is, rather than relying solely on ground-truth labels, we also use the soft predictions from a teacher network pre-trained on the complete data. By integrating KD into training, we demonstrate significant improvement across datasets, pruning methods, and on all pruning fractions. We first establish a theoretical motivation for employing self-distillation to improve training on pruned data. Then, we empirically make a compelling and highly practical observation: using KD, simple random pruning is c
    
[^5]: CFRet-DVQA：粗到精检索和高效调优用于文档视觉问答

    CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering

    [https://arxiv.org/abs/2403.00816](https://arxiv.org/abs/2403.00816)

    该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。

    

    文档视觉问答（DVQA）是一个涉及根据图像内容回答查询的任务。现有工作仅限于定位单页内的信息，不支持跨页面问答交互。此外，对模型输入的标记长度限制可能导致与答案相关的部分被截断。在本研究中，我们引入了一种简单但有效的方法学，称为CFRet-DVQA，重点放在检索和高效调优上，以有效解决这一关键问题。为此，我们首先从文档中检索与所提问题相关的多个片段。随后，我们利用大型语言模型（LLM）的先进推理能力，通过指导调优进一步增强其性能。该方法使得生成的答案与文档标签的风格相符。实验演示了...

    arXiv:2403.00816v1 Announce Type: cross  Abstract: Document Visual Question Answering (DVQA) is a task that involves responding to queries based on the content of images. Existing work is limited to locating information within a single page and does not facilitate cross-page question-and-answer interaction. Furthermore, the token length limitation imposed on inputs to the model may lead to truncation of segments pertinent to the answer. In this study, we introduce a simple but effective methodology called CFRet-DVQA, which focuses on retrieval and efficient tuning to address this critical issue effectively. For that, we initially retrieve multiple segments from the document that correlate with the question at hand. Subsequently, we leverage the advanced reasoning abilities of the large language model (LLM), further augmenting its performance through instruction tuning. This approach enables the generation of answers that align with the style of the document labels. The experiments demo
    
[^6]: 使用传输的信息瓶颈实现解缠表示学习

    Disentangled Representation Learning with Transmitted Information Bottleneck. (arXiv:2311.01686v1 [cs.CV])

    [http://arxiv.org/abs/2311.01686](http://arxiv.org/abs/2311.01686)

    本研究提出了一种通过引入传输信息瓶颈来实现解缠表示学习的方法。该方法可以在压缩表示信息和保留重要信息之间维持平衡，从而提高模型的稳健性和泛化能力。通过使用贝叶斯网络和变分推断，我们得到了可计算估计的DisTIB。

    

    仅编码与任务相关的原始数据信息，即解缠表示学习，可以极大地提高模型的稳健性和泛化能力。虽然在表示中利用信息理论对信息进行规范化取得了重大进展，但仍存在两个主要挑战：1）表示压缩不可避免地导致性能下降；2）对表示的解缠约束存在复杂的优化问题。针对这些问题，我们引入了传输信息的贝叶斯网络来描述解缠过程中输入和表示之间的相互作用。基于这个框架，我们提出了"DisTIB"（用于解缠表示学习的传输信息瓶颈），一种新的目标函数，用于平衡信息压缩和保留之间的关系。我们采用变分推断来导出DisTIB的可计算估计。

    Encoding only the task-related information from the raw data, \ie, disentangled representation learning, can greatly contribute to the robustness and generalizability of models. Although significant advances have been made by regularizing the information in representations with information theory, two major challenges remain: 1) the representation compression inevitably leads to performance drop; 2) the disentanglement constraints on representations are in complicated optimization. To these issues, we introduce Bayesian networks with transmitted information to formulate the interaction among input and representations during disentanglement. Building upon this framework, we propose \textbf{DisTIB} (\textbf{T}ransmitted \textbf{I}nformation \textbf{B}ottleneck for \textbf{Dis}entangled representation learning), a novel objective that navigates the balance between information compression and preservation. We employ variational inference to derive a tractable estimation for DisTIB. This es
    
[^7]: AutoCLIP: 自动调谐视觉语言模型的零样本分类器

    AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models. (arXiv:2309.16414v1 [cs.CV])

    [http://arxiv.org/abs/2309.16414](http://arxiv.org/abs/2309.16414)

    本研究提出了一种名为AutoCLIP的方法，用于自动调谐视觉语言模型的零样本分类器。AutoCLIP通过为每个提示模板分配图像特定的权重，从而改进了从编码类别描述符推导零样本分类器的方式。

    

    基于视觉语言模型（如CLIP）构建的分类器在广泛的图像分类任务中展现了出色的零样本性能。先前的工作研究了根据提示模板自动创建每个类别的描述符集的不同方式，包括手工设计的模板、从大型语言模型获取的模板以及从随机单词和字符构建的模板。然而，从相应的编码类别描述符导出零样本分类器几乎没有改变：将图像的平均编码类别描述符与编码图像之间的余弦相似度最大化以进行分类。然而，当某些描述符比其他描述符更好地匹配给定图像上的视觉线索时，将所有类别描述符等权重可能不是最优的。在这项工作中，我们提出了一种自动调谐零样本分类器的方法AutoCLIP。AutoCLIP为每个提示模板分配了图像特定的权重，这些权重是从s

    Classifiers built upon vision-language models such as CLIP have shown remarkable zero-shot performance across a broad range of image classification tasks. Prior work has studied different ways of automatically creating descriptor sets for every class based on prompt templates, ranging from manually engineered templates over templates obtained from a large language model to templates built from random words and characters. In contrast, deriving zero-shot classifiers from the respective encoded class descriptors has remained nearly unchanged, that is: classify to the class that maximizes the cosine similarity between its averaged encoded class descriptors and the encoded image. However, weighting all class descriptors equally can be suboptimal when certain descriptors match visual clues on a given image better than others. In this work, we propose AutoCLIP, a method for auto-tuning zero-shot classifiers. AutoCLIP assigns to each prompt template per-image weights, which are derived from s
    
[^8]: 自监督可扩展深度压缩感知

    Self-Supervised Scalable Deep Compressed Sensing. (arXiv:2308.13777v1 [eess.SP])

    [http://arxiv.org/abs/2308.13777](http://arxiv.org/abs/2308.13777)

    本文提出了一种自监督的可扩展深度压缩感知方法，不需要标记的测量-地面真实数据，并且可以处理任意的采样比率和矩阵。该方法包括一个双域损失和四个恢复阶段，通过最大化数据/信息利用率来提高准确性。

    

    压缩感知（CS）是降低采样成本的一种有前景的工具。当前基于深度神经网络（NN）的CS方法在收集标记的测量-地面真实（GT）数据和推广到实际应用方面面临挑战。本文提出了一种新颖的自监督可扩展深度CS方法，包括一个称为SCL的学习方案和一个名为SCNet的网络系列，它不需要GT并且可以处理一旦在部分测量集上训练完毕就可以处理任意的采样比率和矩阵。我们的SCL包含一个双域损失和一个四阶段恢复策略。前者鼓励两个测量部分的交叉一致性以及采样-重构循环一致性，从而最大化数据/信息利用率。后者可以逐步利用外部测量中的常见信号先验和测试样本以及学习的NN的内部特征来提高准确性。

    Compressed sensing (CS) is a promising tool for reducing sampling costs. Current deep neural network (NN)-based CS methods face challenges in collecting labeled measurement-ground truth (GT) data and generalizing to real applications. This paper proposes a novel $\mathbf{S}$elf-supervised s$\mathbf{C}$alable deep CS method, comprising a $\mathbf{L}$earning scheme called $\mathbf{SCL}$ and a family of $\mathbf{Net}$works named $\mathbf{SCNet}$, which does not require GT and can handle arbitrary sampling ratios and matrices once trained on a partial measurement set. Our SCL contains a dual-domain loss and a four-stage recovery strategy. The former encourages a cross-consistency on two measurement parts and a sampling-reconstruction cycle-consistency regarding arbitrary ratios and matrices to maximize data/information utilization. The latter can progressively leverage common signal prior in external measurements and internal characteristics of test samples and learned NNs to improve accur
    
[^9]: 基于补丁化自编码器的视觉异常检测

    Patch-wise Auto-Encoder for Visual Anomaly Detection. (arXiv:2308.00429v1 [cs.CV])

    [http://arxiv.org/abs/2308.00429](http://arxiv.org/abs/2308.00429)

    本论文提出了一种新颖的补丁化自编码器（Patch AE）框架来增强自编码器对异常的重构能力，并在Mvtec AD基准测试中取得了最先进的表现，具有在实际工业应用场景中的潜力。

    

    在没有异常先验的情况下进行异常检测是具有挑战性的。在无监督异常检测领域，传统的自编码器（AE）在仅通过正常图像进行训练时倾向于失败，因为模型将无法正确重构异常图像。相反，我们提出了一种新颖的补丁化自编码器（Patch AE）框架，旨在增强AE对异常的重构能力而不是削弱它。图像的每个补丁都通过相应的空间分布特征向量的学习特征表示进行重构，即补丁化重构，这确保了AE对异常的敏感性。我们的方法简单高效。它在Mvtec AD基准测试中取得了最先进的表现，证明了我们模型的有效性。它在实际工业应用场景中具有巨大潜力。

    Anomaly detection without priors of the anomalies is challenging. In the field of unsupervised anomaly detection, traditional auto-encoder (AE) tends to fail based on the assumption that by training only on normal images, the model will not be able to reconstruct abnormal images correctly. On the contrary, we propose a novel patch-wise auto-encoder (Patch AE) framework, which aims at enhancing the reconstruction ability of AE to anomalies instead of weakening it. Each patch of image is reconstructed by corresponding spatially distributed feature vector of the learned feature representation, i.e., patch-wise reconstruction, which ensures anomaly-sensitivity of AE. Our method is simple and efficient. It advances the state-of-the-art performances on Mvtec AD benchmark, which proves the effectiveness of our model. It shows great potential in practical industrial application scenarios.
    

