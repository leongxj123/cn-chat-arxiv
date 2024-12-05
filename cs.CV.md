# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Addressing Source Scale Bias via Image Warping for Domain Adaptation](https://arxiv.org/abs/2403.12712) | 通过在训练过程中对突出的对象区域进行过采样的自适应注意力处理，以及针对对象区域采样的实例级变形引导，有效减轻域自适应中的源尺度偏差。 |
| [^2] | [Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing](https://arxiv.org/abs/2403.08004) | 本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。 |
| [^3] | [Modeling 3D Infant Kinetics Using Adaptive Graph Convolutional Networks](https://arxiv.org/abs/2402.14400) | 使用数据驱动评估个体动作模式，利用自适应图卷积网络对3D婴儿动力学进行建模，相较于传统机器学习取得了改进。 |
| [^4] | [Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID](https://arxiv.org/abs/2402.00672) | 该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。 |
| [^5] | [One Step Learning, One Step Review.](http://arxiv.org/abs/2401.10962) | 本文提出了一种基于权重回滚的微调方法OLOR，通过结合优化器和权重回滚项，解决了完全微调方法中的知识遗忘问题，并在各种任务上提高了微调性能。 |
| [^6] | [FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus.](http://arxiv.org/abs/2310.11178) | FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。 |
| [^7] | [OpenDriver: an open-road driver state detection dataset.](http://arxiv.org/abs/2304.04203) | OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。 |

# 详细

[^1]: 通过图像变形解决域自适应中的源尺度偏差问题

    Addressing Source Scale Bias via Image Warping for Domain Adaptation

    [https://arxiv.org/abs/2403.12712](https://arxiv.org/abs/2403.12712)

    通过在训练过程中对突出的对象区域进行过采样的自适应注意力处理，以及针对对象区域采样的实例级变形引导，有效减轻域自适应中的源尺度偏差。

    

    在视觉识别中，由于真实场景数据集中对象和图像大小分布的不平衡，尺度偏差是一个关键挑战。传统解决方案包括注入尺度不变性先验、在训练过程中对数据集在不同尺度进行过采样，或者在推断时调整尺度。虽然这些策略在一定程度上减轻了尺度偏差，但它们在跨多样化数据集时的适应能力有限。此外，它们会增加训练过程的计算负载和推断过程的延迟。在这项工作中，我们使用自适应的注意力处理——通过在训练过程中就地扭曲图像来对突出的对象区域进行过采样。我们发现，通过改变源尺度分布可以改善主干特征，我们开发了一个面向对象区域采样的实例级变形引导，以减轻域自适应中的源尺度偏差。我们的方法提高了对地理、光照和天气条件的适应性。

    arXiv:2403.12712v1 Announce Type: cross  Abstract: In visual recognition, scale bias is a key challenge due to the imbalance of object and image size distribution inherent in real scene datasets. Conventional solutions involve injecting scale invariance priors, oversampling the dataset at different scales during training, or adjusting scale at inference. While these strategies mitigate scale bias to some extent, their ability to adapt across diverse datasets is limited. Besides, they increase computational load during training and latency during inference. In this work, we use adaptive attentional processing -- oversampling salient object regions by warping images in-place during training. Discovering that shifting the source scale distribution improves backbone features, we developed a instance-level warping guidance aimed at object region sampling to mitigate source scale bias in domain adaptation. Our approach improves adaptation across geographies, lighting and weather conditions, 
    
[^2]: Pix2Pix-OnTheFly: 利用LLMs进行指导图像编辑

    Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing

    [https://arxiv.org/abs/2403.08004](https://arxiv.org/abs/2403.08004)

    本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。

    

    众所周知，最近结合语言处理和图像处理的研究引起了广泛关注，本文提出了一种全新的方法，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，而无需预备工作，证明了该方法的有效性和竞争力。

    arXiv:2403.08004v1 Announce Type: cross  Abstract: The combination of language processing and image processing keeps attracting increased interest given recent impressive advances that leverage the combined strengths of both domains of research. Among these advances, the task of editing an image on the basis solely of a natural language instruction stands out as a most challenging endeavour. While recent approaches for this task resort, in one way or other, to some form of preliminary preparation, training or fine-tuning, this paper explores a novel approach: We propose a preparation-free method that permits instruction-guided image editing on the fly. This approach is organized along three steps properly orchestrated that resort to image captioning and DDIM inversion, followed by obtaining the edit direction embedding, followed by image editing proper. While dispensing with preliminary preparation, our approach demonstrates to be effective and competitive, outperforming recent, state 
    
[^3]: 使用自适应图卷积网络对3D婴儿动力学进行建模

    Modeling 3D Infant Kinetics Using Adaptive Graph Convolutional Networks

    [https://arxiv.org/abs/2402.14400](https://arxiv.org/abs/2402.14400)

    使用数据驱动评估个体动作模式，利用自适应图卷积网络对3D婴儿动力学进行建模，相较于传统机器学习取得了改进。

    

    可靠的婴儿神经发育评估方法对于早期发现可能需要及时干预的医学问题至关重要。自发的运动活动，即“动力学”，被证明可提供一个强有力的预测未来神经发育的替代性测量。然而，它的评估在很大程度上是定性和主观的，侧重于对通过视觉识别的特定年龄手势的描述。在这里，我们采用了一种替代方法，根据数据驱动评估个体动作模式来预测婴儿神经发育成熟。我们利用处理过的3D婴儿视频录像进行姿势估计，提取解剖标志物的时空系列，并应用自适应图卷积网络来预测实际年龄。我们展示了我们的数据驱动方法相对于基于手动设计特征的传统机器学习基线取得了改进。

    arXiv:2402.14400v1 Announce Type: cross  Abstract: Reliable methods for the neurodevelopmental assessment of infants are essential for early detection of medical issues that may need prompt interventions. Spontaneous motor activity, or `kinetics', is shown to provide a powerful surrogate measure of upcoming neurodevelopment. However, its assessment is by and large qualitative and subjective, focusing on visually identified, age-specific gestures. Here, we follow an alternative approach, predicting infants' neurodevelopmental maturation based on data-driven evaluation of individual motor patterns. We utilize 3D video recordings of infants processed with pose-estimation to extract spatio-temporal series of anatomical landmarks, and apply adaptive graph convolutional networks to predict the actual age. We show that our data-driven approach achieves improvement over traditional machine learning baselines based on manually engineered features.
    
[^4]: 探索用于无监督可见-红外人物重新识别的均质和异质一致标签关联

    Exploring Homogeneous and Heterogeneous Consistent Label Associations for Unsupervised Visible-Infrared Person ReID

    [https://arxiv.org/abs/2402.00672](https://arxiv.org/abs/2402.00672)

    该论文提出了一种同时考虑均质和异质实例级别结构，构建高质量跨模态标签关联的模态统一标签传输方法，用于无监督可见-红外人物重新识别。

    

    无监督可见-红外人物重新识别（USL-VI-ReID）旨在无需注释从不同模态中检索相同身份的行人图像。之前的研究侧重于建立跨模态的伪标签关联以弥合模态间的差异，但忽略了在伪标签空间中保持实例级别的均质和异质一致性，导致关联粗糙。为此，我们引入了一个模态统一标签传输（MULT）模块，同时考虑了均质和异质细粒度实例级结构，生成高质量的跨模态标签关联。它建模了均质和异质的关联性，利用它们定义伪标签的不一致性，然后最小化这种不一致性，从而维持了跨模态的对齐并保持了内部模态结构的一致性。此外，还有一个简单易用的在线交叉记忆标签引用模块。

    Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to retrieve pedestrian images of the same identity from different modalities without annotations. While prior work focuses on establishing cross-modality pseudo-label associations to bridge the modality-gap, they ignore maintaining the instance-level homogeneous and heterogeneous consistency in pseudo-label space, resulting in coarse associations. In response, we introduce a Modality-Unified Label Transfer (MULT) module that simultaneously accounts for both homogeneous and heterogeneous fine-grained instance-level structures, yielding high-quality cross-modality label associations. It models both homogeneous and heterogeneous affinities, leveraging them to define the inconsistency for the pseudo-labels and then minimize it, leading to pseudo-labels that maintain alignment across modalities and consistency within intra-modality structures. Additionally, a straightforward plug-and-play Online Cross-memory Label Ref
    
[^5]: 一步学习，一步评审

    One Step Learning, One Step Review. (arXiv:2401.10962v1 [cs.CV])

    [http://arxiv.org/abs/2401.10962](http://arxiv.org/abs/2401.10962)

    本文提出了一种基于权重回滚的微调方法OLOR，通过结合优化器和权重回滚项，解决了完全微调方法中的知识遗忘问题，并在各种任务上提高了微调性能。

    

    随着预训练视觉模型的兴起，视觉微调已经引起了广泛关注。当前主流的方法——完全微调，存在知识遗忘的问题，因为它只专注于拟合下游训练集。在本文中，我们提出了一种新颖的基于权重回滚的微调方法，称为OLOR（一步学习，一步评审）。OLOR将微调与优化器相结合，将权重回滚项加入到每个步骤的权重更新项中。这确保了上游和下游模型的权重范围的一致性，有效地减轻了知识遗忘问题，并增强了微调性能。此外，我们提出了一种逐层惩罚方法，通过 penalty decay 和不同的衰减率来调整层的权重回滚程度，以适应不同的下游任务。通过在图像分类、目标检测、语义分割和实例分割等各种任务上进行大量实验证明，我们的方法提高了微调的性能。

    Visual fine-tuning has garnered significant attention with the rise of pre-trained vision models. The current prevailing method, full fine-tuning, suffers from the issue of knowledge forgetting as it focuses solely on fitting the downstream training set. In this paper, we propose a novel weight rollback-based fine-tuning method called OLOR (One step Learning, One step Review). OLOR combines fine-tuning with optimizers, incorporating a weight rollback term into the weight update term at each step. This ensures consistency in the weight range of upstream and downstream models, effectively mitigating knowledge forgetting and enhancing fine-tuning performance. In addition, a layer-wise penalty is presented to employ penalty decay and the diversified decay rate to adjust the weight rollback levels of layers for adapting varying downstream tasks. Through extensive experiments on various tasks such as image classification, object detection, semantic segmentation, and instance segmentation, we
    
[^6]: FocDepthFormer: 使用LSTM的Transformer用于从焦点进行深度估计

    FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus. (arXiv:2310.11178v1 [cs.CV])

    [http://arxiv.org/abs/2310.11178](http://arxiv.org/abs/2310.11178)

    FocDepthFormer是一种基于Transformer和LSTM的网络，用于从焦点进行深度估计。通过Transformer的自注意力和LSTM的集成，该方法能够学习更多有信息的特征，并且具有对任意长度堆栈的泛化能力。

    

    从焦点堆栈进行深度估计是一个基本的计算机视觉问题，旨在通过图像堆栈中的焦点/离焦线索推断深度。大多数现有方法通过在一组固定的图像堆栈上应用二维或三维卷积神经网络（CNNs）来处理此问题，以在图像和堆栈之间学习特征。由于CNN的局部性质，它们的性能受到限制，并且它们被限制在处理在训练和推断中一致的固定数量的堆栈上，从而限制了对任意长度堆栈的泛化能力。为了解决上述限制，我们开发了一种新颖的基于Transformer的网络，FocDepthFormer，主要由带有LSTM模块和CNN解码器的Transformer组成。Transformer中的自注意力通过隐含非局部交叉参考能够学习更多有信息的特征。LSTM模块被学习用于将表示集成到具有任意图像的堆栈中。为了直接捕获低级特征

    Depth estimation from focal stacks is a fundamental computer vision problem that aims to infer depth from focus/defocus cues in the image stacks. Most existing methods tackle this problem by applying convolutional neural networks (CNNs) with 2D or 3D convolutions over a set of fixed stack images to learn features across images and stacks. Their performance is restricted due to the local properties of the CNNs, and they are constrained to process a fixed number of stacks consistent in train and inference, limiting the generalization to the arbitrary length of stacks. To handle the above limitations, we develop a novel Transformer-based network, FocDepthFormer, composed mainly of a Transformer with an LSTM module and a CNN decoder. The self-attention in Transformer enables learning more informative features via an implicit non-local cross reference. The LSTM module is learned to integrate the representations across the stack with arbitrary images. To directly capture the low-level featur
    
[^7]: OpenDriver: 一份开放路况驾驶员状态检测数据集

    OpenDriver: an open-road driver state detection dataset. (arXiv:2304.04203v1 [cs.AI])

    [http://arxiv.org/abs/2304.04203](http://arxiv.org/abs/2304.04203)

    OpenDriver是一份旨在解决现有驾驶员生理数据集存在问题的开放路况驾驶员状态检测数据集，包含六轴惯性信号和心电图信号两种模态的数据，可用于驾驶员受损检测和生物识别数据识别。

    

    在现代社会中，道路安全严重依赖于驾驶员的心理和生理状态。疲劳、昏昏欲睡和压力等负面因素会影响驾驶员的反应时间和决策能力，导致交通事故的发生率增加。在众多的驾驶员行为监测研究中，可穿戴生理测量是一种实时监测驾驶员状态的方法。然而，目前在开放道路场景下，缺少驾驶员生理数据集，已有的数据集存在信号质量差、样本量小和数据收集时间短等问题。因此，本文设计并描述了一种大规模多模态驾驶数据集，用于驾驶员受损检测和生物识别数据识别。该数据集包含两种驾驶信号模态：六轴惯性信号和心电图（ECG）信号，这些信号是在100多名驾驶员遵循相同路线行驶时记录的。

    In modern society, road safety relies heavily on the psychological and physiological state of drivers. Negative factors such as fatigue, drowsiness, and stress can impair drivers' reaction time and decision making abilities, leading to an increased incidence of traffic accidents. Among the numerous studies for impaired driving detection, wearable physiological measurement is a real-time approach to monitoring a driver's state. However, currently, there are few driver physiological datasets in open road scenarios and the existing datasets suffer from issues such as poor signal quality, small sample sizes, and short data collection periods. Therefore, in this paper, a large-scale multimodal driving dataset for driver impairment detection and biometric data recognition is designed and described. The dataset contains two modalities of driving signals: six-axis inertial signals and electrocardiogram (ECG) signals, which were recorded while over one hundred drivers were following the same ro
    

