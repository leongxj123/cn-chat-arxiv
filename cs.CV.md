# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training](https://arxiv.org/abs/2403.09948) | RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。 |
| [^2] | [Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning](https://arxiv.org/abs/2402.15761) | Res-VMamba利用具有选择性状态空间模型和深度残差学习，提供了比Transformer结构更出色的性能和计算效率，是食品细粒度分类中的最新技术。 |
| [^3] | [Is my Data in your AI Model? Membership Inference Test with Application to Face Images](https://arxiv.org/abs/2402.09225) | This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy. |
| [^4] | [Less is More: Fewer Interpretable Region via Submodular Subset Selection](https://arxiv.org/abs/2402.09164) | 本论文将图像归属问题重新建模为次模子集选择问题，通过使用更少的区域来增强模型的解释性，解决了现有归属解决方案面临的不准确区域和预测错误样本的问题。 |
| [^5] | [FRS-Nets: Fourier Parameterized Rotation and Scale Equivariant Networks for Retinal Vessel Segmentation.](http://arxiv.org/abs/2309.15638) | 本研究提出了一种名为FRS-Nets的新型卷积神经网络方法，利用傅里叶参数化实现了对旋转和尺度的等变性，从而提升了视网膜血管分割的准确性。通过在U-Net和Iter-Net中替换传统卷积滤波器，实现了更好的分割效果。 |
| [^6] | [EgoPoser: Robust Real-Time Ego-Body Pose Estimation in Large Scenes.](http://arxiv.org/abs/2308.06493) | 本文提出了EgoPoser，一种能够在大场景中鲁棒地实时估计自我身体姿势的方法。通过重新思考输入表示、引入新的运动分解方法以及建模身体姿势，EgoPoser在定性和定量上均表现优于现有方法，并具有较高的推理速度。 |
| [^7] | [Video alignment using unsupervised learning of local and global features.](http://arxiv.org/abs/2304.06841) | 本文提出了一种无需训练的视频对齐方法，利用全局和局部特征将帧转化为时间序列并使用对角化动态时间规整算法进行对齐。 |

# 详细

[^1]: RadCLIP: 通过对比语言图像预训练增强放射学图像分析

    RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training

    [https://arxiv.org/abs/2403.09948](https://arxiv.org/abs/2403.09948)

    RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。

    

    arXiv:2403.09948v1 公告类型: 跨领域  摘要: 人工智能（AI）与放射学的整合标志着医学诊断领域的变革时代。视觉基础模型已被采用来增强放射学图像分析。然而，放射学图像的独特复杂性，包括对2D和3D放射学数据的解读，带来了现有模型无法充分应对的挑战，因为这些模型是在通用非医学图像上训练的。为了弥合这一差距，并充分利用医学成像所需的诊断精度，我们引入了RadCLIP：一种开创性的跨模态基础模型，利用对比语言图像预训练（CLIP）来改进放射学图像分析。RadCLIP包含一种新颖的3D切片池化机制，专为体积图像分析定制，使用了丰富多样的放射学图像-文本对数据集进行训练。我们的评估表明，RadCLIP能有效地对齐放射学图像

    arXiv:2403.09948v1 Announce Type: cross  Abstract: The integration of artificial intelligence (AI) with radiology has marked a transformative era in medical diagnostics. Vision foundation models have been adopted to enhance radiologic imaging analysis. However, the distinct complexities of radiological imaging, including the interpretation of 2D and 3D radiological data, pose unique challenges that existing models, trained on general non-medical images, fail to address adequately. To bridge this gap and capitalize on the diagnostic precision required in medical imaging, we introduce RadCLIP: a pioneering cross-modal foundational model that harnesses Contrastive Language-Image Pre-training (CLIP) to refine radiologic image analysis. RadCLIP incorporates a novel 3D slice pooling mechanism tailored for volumetric image analysis and is trained using a comprehensive and diverse dataset of radiologic image-text pairs. Our evaluations demonstrate that RadCLIP effectively aligns radiological i
    
[^2]: 使用具有深度残差学习的选择性状态空间模型进行细粒度食品类别视觉分类的Res-VMamba

    Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning

    [https://arxiv.org/abs/2402.15761](https://arxiv.org/abs/2402.15761)

    Res-VMamba利用具有选择性状态空间模型和深度残差学习，提供了比Transformer结构更出色的性能和计算效率，是食品细粒度分类中的最新技术。

    

    食品分类是发展食品视觉任务的基础，并在计算营养学这一新兴领域中发挥着关键作用。由于食物的复杂性需要细粒度分类，最近的学术研究主要修改卷积神经网络(CNNs)和/或视觉变压器(ViTs)来执行食品类别分类。然而，为了学习细粒度特征，CNN骨干需要额外的结构设计，而包含自注意力模块的ViT具有更高的计算复杂性。最近推出的新的序列状态空间(S4)模型，通过选择机制和与扫描(S6)的计算，俗称为Mamba，相较于变压器架构展示了卓越的性能和计算效率。将Mamba机制整合到图像任务(如分类)中的VMamba模型目前建立了最先进技术

    arXiv:2402.15761v1 Announce Type: cross  Abstract: Food classification is the foundation for developing food vision tasks and plays a key role in the burgeoning field of computational nutrition. Due to the complexity of food requiring fine-grained classification, recent academic research mainly modifies Convolutional Neural Networks (CNNs) and/or Vision Transformers (ViTs) to perform food category classification. However, to learn fine-grained features, the CNN backbone needs additional structural design, whereas ViT, containing the self-attention module, has increased computational complexity. In recent months, a new Sequence State Space (S4) model, through a Selection mechanism and computation with a Scan (S6), colloquially termed Mamba, has demonstrated superior performance and computation efficiency compared to the Transformer architecture. The VMamba model, which incorporates the Mamba mechanism into image tasks (such as classification), currently establishes the state-of-the-art 
    
[^3]: 我的数据在你的AI模型中吗？通过应用于人脸图像的成员推断测试

    Is my Data in your AI Model? Membership Inference Test with Application to Face Images

    [https://arxiv.org/abs/2402.09225](https://arxiv.org/abs/2402.09225)

    This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy.

    

    这篇论文介绍了成员推断测试（MINT），一种用于经验性评估特定数据是否被用于训练人工智能（AI）模型的新方法。具体而言，我们提出了两种新颖的MINT架构，旨在学习在经过审计的模型暴露于其训练过程中使用的数据时出现的不同激活模式。第一个架构基于多层感知机（MLP）网络，第二个基于卷积神经网络（CNN）。所提出的MINT架构在具有挑战性的人脸识别任务上进行评估，考虑了三种最先进的人脸识别模型。使用六个公开可用的数据库进行实验，总共包含超过2200万张人脸图像。根据可用的AI模型测试的上下文，考虑了不同的实验场景。有希望的结果达到了90%的准确率。

    arXiv:2402.09225v1 Announce Type: cross Abstract: This paper introduces the Membership Inference Test (MINT), a novel approach that aims to empirically assess if specific data was used during the training of Artificial Intelligence (AI) models. Specifically, we propose two novel MINT architectures designed to learn the distinct activation patterns that emerge when an audited model is exposed to data used during its training process. The first architecture is based on a Multilayer Perceptron (MLP) network and the second one is based on Convolutional Neural Networks (CNNs). The proposed MINT architectures are evaluated on a challenging face recognition task, considering three state-of-the-art face recognition models. Experiments are carried out using six publicly available databases, comprising over 22 million face images in total. Also, different experimental scenarios are considered depending on the context available of the AI model to test. Promising results, up to 90% accuracy, are a
    
[^4]: 简约即是美：通过次模子集选择减少可解释区域

    Less is More: Fewer Interpretable Region via Submodular Subset Selection

    [https://arxiv.org/abs/2402.09164](https://arxiv.org/abs/2402.09164)

    本论文将图像归属问题重新建模为次模子集选择问题，通过使用更少的区域来增强模型的解释性，解决了现有归属解决方案面临的不准确区域和预测错误样本的问题。

    

    图像归属算法旨在确定与模型决策高度相关的重要区域。尽管现有的归属解决方案可以有效地给目标元素分配重要性，但仍面临以下挑战：1）现有的归属方法生成不准确的小区域，从而误导正确归属的方向；2）模型无法为预测错误的样本产生良好的归属结果。为了解决上述挑战，本文将上述图像归属问题重新建模为次模子集选择问题，旨在使用更少的区域增强模型的可解释性。为了解决对局部区域的关注不足，我们构造了一个新的次模函数来发现更准确的精细解释区域。为了增强所有样本的归属效果，我们还对子区域选择施加了四个不同的约束，即置信度，

    arXiv:2402.09164v1 Announce Type: cross Abstract: Image attribution algorithms aim to identify important regions that are highly relevant to model decisions. Although existing attribution solutions can effectively assign importance to target elements, they still face the following challenges: 1) existing attribution methods generate inaccurate small regions thus misleading the direction of correct attribution, and 2) the model cannot produce good attribution results for samples with wrong predictions. To address the above challenges, this paper re-models the above image attribution problem as a submodular subset selection problem, aiming to enhance model interpretability using fewer regions. To address the lack of attention to local regions, we construct a novel submodular function to discover more accurate fine-grained interpretation regions. To enhance the attribution effect for all samples, we also impose four different constraints on the selection of sub-regions, i.e., confidence, 
    
[^5]: FRS-Nets: Fourier参数化的旋转和尺度等变网络用于视网膜血管分割

    FRS-Nets: Fourier Parameterized Rotation and Scale Equivariant Networks for Retinal Vessel Segmentation. (arXiv:2309.15638v1 [eess.IV])

    [http://arxiv.org/abs/2309.15638](http://arxiv.org/abs/2309.15638)

    本研究提出了一种名为FRS-Nets的新型卷积神经网络方法，利用傅里叶参数化实现了对旋转和尺度的等变性，从而提升了视网膜血管分割的准确性。通过在U-Net和Iter-Net中替换传统卷积滤波器，实现了更好的分割效果。

    

    具有平移等变性的卷积神经网络（CNNs）在视网膜血管分割中取得了巨大的成功。然而，CNNs没有对血管形态的其他对称性进行建模，例如旋转和尺度对称性。为了在CNNs中嵌入更多等变性并满足视网膜血管分割的准确性要求，我们构建了一种新颖的卷积算子（FRS-Conv），它是傅里叶参数化的，并且对旋转和缩放等变。具体地，我们首先采用一种新的参数化方案，使卷积滤波器能够以高精度任意进行变换。其次，我们导出了旋转和尺度等变卷积映射的公式。最后，我们根据提出的公式构建了FRS-Conv，并将U-Net和Iter-Net中的传统卷积滤波器替换为FRS-Conv（FRS-Nets）。我们忠实地复现了所有对比方法，并在三个公共数据集上进行了全面实验。

    With translation equivariance, convolution neural networks (CNNs) have achieved great success in retinal vessel segmentation. However, some other symmetries of the vascular morphology are not characterized by CNNs, such as rotation and scale symmetries. To embed more equivariance into CNNs and achieve the accuracy requirement for retinal vessel segmentation, we construct a novel convolution operator (FRS-Conv), which is Fourier parameterized and equivariant to rotation and scaling. Specifically, we first adopt a new parameterization scheme, which enables convolutional filters to arbitrarily perform transformations with high accuracy. Secondly, we derive the formulations for the rotation and scale equivariant convolution mapping. Finally, we construct FRS-Conv following the proposed formulations and replace the traditional convolution filters in U-Net and Iter-Net with FRS-Conv (FRS-Nets). We faithfully reproduce all compared methods and conduct comprehensive experiments on three public
    
[^6]: EgoPoser：大场景下鲁棒的实时自我身体姿势估计

    EgoPoser: Robust Real-Time Ego-Body Pose Estimation in Large Scenes. (arXiv:2308.06493v1 [cs.CV])

    [http://arxiv.org/abs/2308.06493](http://arxiv.org/abs/2308.06493)

    本文提出了EgoPoser，一种能够在大场景中鲁棒地实时估计自我身体姿势的方法。通过重新思考输入表示、引入新的运动分解方法以及建模身体姿势，EgoPoser在定性和定量上均表现优于现有方法，并具有较高的推理速度。

    

    头部和手部姿势仅通过完整身体自我姿势估计已成为研究的一个热点领域，以为头戴式平台上的虚拟角色表达提供动力。然而，现有方法过于依赖数据集记录时的运动捕捉空间的限制，同时假设连续捕捉关节运动和均匀身体尺寸。在本文中，我们提出了EgoPoser，通过以下方式克服了这些限制：1）重新思考基于头戴式平台的自我姿势估计的输入表示，并引入一种新的运动分解方法来预测与全局位置无关的完整身体姿势，2）从头戴式设备视野内的间歇性手部姿势跟踪中鲁棒地建模身体姿势，3）针对不同用户的各种身体尺寸进行通用化推广。我们的实验表明，EgoPoser在定性和定量上优于现有的方法，并保持较高的推理速度。

    Full-body ego-pose estimation from head and hand poses alone has become an active area of research to power articulate avatar representation on headset-based platforms. However, existing methods over-rely on the confines of the motion-capture spaces in which datasets were recorded, while simultaneously assuming continuous capture of joint motions and uniform body dimensions. In this paper, we propose EgoPoser, which overcomes these limitations by 1) rethinking the input representation for headset-based ego-pose estimation and introducing a novel motion decomposition method that predicts full-body pose independent of global positions, 2) robustly modeling body pose from intermittent hand position and orientation tracking only when inside a headset's field of view, and 3) generalizing across various body sizes for different users. Our experiments show that EgoPoser outperforms state-of-the-art methods both qualitatively and quantitatively, while maintaining a high inference speed of over
    
[^7]: 无监督学习局部和全局特征用于视频对齐

    Video alignment using unsupervised learning of local and global features. (arXiv:2304.06841v1 [cs.CV])

    [http://arxiv.org/abs/2304.06841](http://arxiv.org/abs/2304.06841)

    本文提出了一种无需训练的视频对齐方法，利用全局和局部特征将帧转化为时间序列并使用对角化动态时间规整算法进行对齐。

    

    本文致力于解决视频对齐的问题，即匹配包含相似活动的一对视频的帧。视频对齐的主要挑战在于，尽管两个视频之间的执行过程和外观有所不同，但仍需要建立精确的对应关系。我们提出了一种使用帧的全局和局部特征进行对齐的无监督方法。特别地，我们利用人物检测、姿态估计和VGG网络三种机器视觉工具为每个视频帧引入有效的特征。然后对这些特征进行处理和组合以构建代表视频的多维时间序列。使用一种名为对角化动态时间规整的新版本（Diagonalized Dynamic Time Warping, DDTW）对生成的时间序列进行对齐。我们的方法的主要优点在于不需要任何训练，因此适用于任何新类型的活动而无需处理。

    In this paper, we tackle the problem of video alignment, the process of matching the frames of a pair of videos containing similar actions. The main challenge in video alignment is that accurate correspondence should be established despite the differences in the execution processes and appearances between the two videos. We introduce an unsupervised method for alignment that uses global and local features of the frames. In particular, we introduce effective features for each video frame by means of three machine vision tools: person detection, pose estimation, and VGG network. Then the features are processed and combined to construct a multidimensional time series that represent the video. The resulting time series are used to align videos of the same actions using a novel version of dynamic time warping named Diagonalized Dynamic Time Warping(DDTW). The main advantage of our approach is that no training is required, which makes it applicable for any new type of action without any need
    

