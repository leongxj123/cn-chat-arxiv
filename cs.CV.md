# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Data for Robust Stroke Segmentation](https://arxiv.org/abs/2404.01946) | 提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。 |
| [^2] | [Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning](https://arxiv.org/abs/2404.00785) | 本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。 |
| [^3] | [Fine Structure-Aware Sampling: A New Sampling Training Scheme for Pixel-Aligned Implicit Models in Single-View Human Reconstruction](https://arxiv.org/abs/2402.19197) | FSS是一种新的用于单视图人体重建中像素对齐隐式模型的采样训练方案，通过主动适应表面的厚度和复杂性，以及利用样本点的法线来改善结果，同时引入网格厚度损失信号来进一步改进训练过程。 |
| [^4] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |
| [^5] | [Hierarchical Randomized Smoothing.](http://arxiv.org/abs/2310.16221) | 分层随机平滑是一种在复杂数据上进行鲁棒性认证的解决方案，通过只在一个对象的子集上添加随机噪声，以更有针对性的方式提供了更强的鲁棒性保证和高准确性。 |
| [^6] | [Evaluation of Environmental Conditions on Object Detection using Oriented Bounding Boxes for AR Applications.](http://arxiv.org/abs/2306.16798) | 本研究提出了一种使用定向边界框和深度网络进行物体检测的新方法，通过在不同环境条件下对两个数据集的评估发现，该方法在处理小物体时能够提高性能和准确性。 |
| [^7] | [MixMask: Revisiting Masking Strategy for Siamese ConvNets.](http://arxiv.org/abs/2210.11456) | 本文提出了一种新的填充式遮盖策略MixMask，在Siamese ConvNets中实现遮盖和对比学习目标的匹配，提高了Siamese ConvNets的性能并在多个基准测试中实现了最先进的结果。 |

# 详细

[^1]: 用于鲁棒性中风分割的合成数据

    Synthetic Data for Robust Stroke Segmentation

    [https://arxiv.org/abs/2404.01946](https://arxiv.org/abs/2404.01946)

    提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。

    

    arXiv:2404.01946v1 公告类型：交叉 摘要：目前基于深度学习的神经影像语义分割需要高分辨率扫描和大量注释数据集，这给临床适用性带来了显著障碍。我们提出了一种新颖的合成框架，用于病变分割任务，扩展了已建立的SynthSeg方法的能力，以适应具有病变特定增强策略的大型异质病变。我们的方法使用从健康和中风数据集派生的标签映射训练深度学习模型，在这里演示了UNet架构，促进了健康组织和病理病变的分割，而无需特定于序列的训练数据。针对领域内和领域外（OOD）数据集进行评估，我们的框架表现出鲁棒性能，与训练领域内的当前方法相媲美，并在OOD数据上显着优于它们。这一贡献有望推动医学...

    arXiv:2404.01946v1 Announce Type: cross  Abstract: Deep learning-based semantic segmentation in neuroimaging currently requires high-resolution scans and extensive annotated datasets, posing significant barriers to clinical applicability. We present a novel synthetic framework for the task of lesion segmentation, extending the capabilities of the established SynthSeg approach to accommodate large heterogeneous pathologies with lesion-specific augmentation strategies. Our method trains deep learning models, demonstrated here with the UNet architecture, using label maps derived from healthy and stroke datasets, facilitating the segmentation of both healthy tissue and pathological lesions without sequence-specific training data. Evaluated against in-domain and out-of-domain (OOD) datasets, our framework demonstrates robust performance, rivaling current methods within the training domain and significantly outperforming them on OOD data. This contribution holds promise for advancing medical
    
[^2]: 解开海马形状变异之谜：利用对比学习的图变分自动编码器研究神经系统疾病

    Disentangling Hippocampal Shape Variations: A Study of Neurological Disorders Using Graph Variational Autoencoder with Contrastive Learning

    [https://arxiv.org/abs/2404.00785](https://arxiv.org/abs/2404.00785)

    本研究利用图变分自动编码器和对比学习解开神经系统疾病中海马形状变异的关键潜变量，超越了其他先进方法在解开能力上的表现。

    

    本文提出了一项综合研究，专注于在神经系统疾病背景下从扩散张量成像（DTI）数据集中解开海马形状变异。借助增强的监督对比学习图变分自动编码器（VAE），我们的方法旨在通过区分代表年龄和是否患病的两个不同潜变量来提高解释性。在我们的消融研究中，我们调查了一系列VAE架构和对比损失函数，展示了我们方法增强的解开能力。这个评估使用了来自DTI海马数据集的合成3D环形网格数据和真实的3D海马网格数据集。我们的监督解开模型在解开分数方面优于几种最先进的方法，如属性和引导VAE。我们的模型可以区分不同年龄组和疾病状况。

    arXiv:2404.00785v1 Announce Type: cross  Abstract: This paper presents a comprehensive study focused on disentangling hippocampal shape variations from diffusion tensor imaging (DTI) datasets within the context of neurological disorders. Leveraging a Graph Variational Autoencoder (VAE) enhanced with Supervised Contrastive Learning, our approach aims to improve interpretability by disentangling two distinct latent variables corresponding to age and the presence of diseases. In our ablation study, we investigate a range of VAE architectures and contrastive loss functions, showcasing the enhanced disentanglement capabilities of our approach. This evaluation uses synthetic 3D torus mesh data and real 3D hippocampal mesh datasets derived from the DTI hippocampal dataset. Our supervised disentanglement model outperforms several state-of-the-art (SOTA) methods like attribute and guided VAEs in terms of disentanglement scores. Our model distinguishes between age groups and disease status in pa
    
[^3]: 细结构感知采样: 一种新的用于单视图人体重建中像素对齐隐式模型的采样训练方案

    Fine Structure-Aware Sampling: A New Sampling Training Scheme for Pixel-Aligned Implicit Models in Single-View Human Reconstruction

    [https://arxiv.org/abs/2402.19197](https://arxiv.org/abs/2402.19197)

    FSS是一种新的用于单视图人体重建中像素对齐隐式模型的采样训练方案，通过主动适应表面的厚度和复杂性，以及利用样本点的法线来改善结果，同时引入网格厚度损失信号来进一步改进训练过程。

    

    像素对齐的隐式模型，如PIFu、PIFuHD和ICON，用于单视图着装人体重建。这些模型需要使用采样训练方案进行训练。现有的采样训练方案要么无法捕捉薄表面（如耳朵、手指），要么会导致重建网格中的噪声伪影。为解决这些问题，我们引入了细结构感知采样（FSS），这是一种新的用于单视图人体重建中训练像素对齐隐式模型的采样训练方案。FSS通过主动适应表面的厚度和复杂性来解决前述问题。此外，与现有的采样训练方案不同，FSS显示了如何利用样本点的法线在训练过程中提高结果。最后，为进一步改进训练过程，FSS提出了一个用于像素对齐隐式模型的网格厚度损失信号。这使得在训练过程中利用法线变得计算上可行。

    arXiv:2402.19197v1 Announce Type: cross  Abstract: Pixel-aligned implicit models, such as PIFu, PIFuHD, and ICON, are used for single-view clothed human reconstruction. These models need to be trained using a sampling training scheme. Existing sampling training schemes either fail to capture thin surfaces (e.g. ears, fingers) or cause noisy artefacts in reconstructed meshes. To address these problems, we introduce Fine Structured-Aware Sampling (FSS), a new sampling training scheme to train pixel-aligned implicit models for single-view human reconstruction. FSS resolves the aforementioned problems by proactively adapting to the thickness and complexity of surfaces. In addition, unlike existing sampling training schemes, FSS shows how normals of sample points can be capitalized in the training process to improve results. Lastly, to further improve the training process, FSS proposes a mesh thickness loss signal for pixel-aligned implicit models. It becomes computationally feasible to int
    
[^4]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    
[^5]: 分层随机平滑

    Hierarchical Randomized Smoothing. (arXiv:2310.16221v1 [cs.LG])

    [http://arxiv.org/abs/2310.16221](http://arxiv.org/abs/2310.16221)

    分层随机平滑是一种在复杂数据上进行鲁棒性认证的解决方案，通过只在一个对象的子集上添加随机噪声，以更有针对性的方式提供了更强的鲁棒性保证和高准确性。

    

    真实世界的数据是复杂的，通常由可分解为多个实体的对象组成（例如，将图像分解为像素，将图形分解为相互连接的节点）。随机平滑是一种强大的框架，可以使模型在其输入的微小变化上具有证明的鲁棒性-通过在分类之前随机添加噪声来保证多数投票的鲁棒性。然而，当对手不是任意干扰整个对象（例如图像），而是对象的某个实体的子集（例如像素）时，通过随机平滑对这种复杂数据进行鲁棒性认证是具有挑战性的。作为解决方案，我们引入了分层随机平滑：我们通过仅在随机选择的实体子集上添加随机噪声来部分平滑对象。通过以比现有方法更有针对性的方式添加噪声，我们获得更强的鲁棒性保证，同时保持高准确性。我们使用不同的噪声分布初始化分层平滑，得到了新的鲁棒性保证。

    Real-world data is complex and often consists of objects that can be decomposed into multiple entities (e.g. images into pixels, graphs into interconnected nodes). Randomized smoothing is a powerful framework for making models provably robust against small changes to their inputs - by guaranteeing robustness of the majority vote when randomly adding noise before classification. Yet, certifying robustness on such complex data via randomized smoothing is challenging when adversaries do not arbitrarily perturb entire objects (e.g. images) but only a subset of their entities (e.g. pixels). As a solution, we introduce hierarchical randomized smoothing: We partially smooth objects by adding random noise only on a randomly selected subset of their entities. By adding noise in a more targeted manner than existing methods we obtain stronger robustness guarantees while maintaining high accuracy. We initialize hierarchical smoothing using different noising distributions, yielding novel robustness
    
[^6]: 评估环境条件对使用定向边界框进行AR应用的物体检测的影响

    Evaluation of Environmental Conditions on Object Detection using Oriented Bounding Boxes for AR Applications. (arXiv:2306.16798v1 [cs.CV])

    [http://arxiv.org/abs/2306.16798](http://arxiv.org/abs/2306.16798)

    本研究提出了一种使用定向边界框和深度网络进行物体检测的新方法，通过在不同环境条件下对两个数据集的评估发现，该方法在处理小物体时能够提高性能和准确性。

    

    增强现实（AR）的目标是将数字内容添加到自然图像和视频中，以创建用户与环境之间的交互体验。场景分析和物体识别在AR中起着至关重要的作用，因为它们必须快速且准确地执行。本研究提出了一种新的方法，利用定向边界框与检测和识别深度网络相结合，以提高性能和处理时间。该方法使用两个数据集进行评估：一个常用于计算机视觉任务的真实图像数据集（DOTA数据集）和一个模拟不同环境、照明和采集条件的合成数据集。评估的重点是小物体，这些物体往往难以检测和识别。结果表明，所提出的方法在大多数测试条件下，对于小物体往往能产生更好的平均精度和更高的准确性。

    The objective of augmented reality (AR) is to add digital content to natural images and videos to create an interactive experience between the user and the environment. Scene analysis and object recognition play a crucial role in AR, as they must be performed quickly and accurately. In this study, a new approach is proposed that involves using oriented bounding boxes with a detection and recognition deep network to improve performance and processing time. The approach is evaluated using two datasets: a real image dataset (DOTA dataset) commonly used for computer vision tasks, and a synthetic dataset that simulates different environmental, lighting, and acquisition conditions. The focus of the evaluation is on small objects, which are difficult to detect and recognise. The results indicate that the proposed approach tends to produce better Average Precision and greater accuracy for small objects in most of the tested conditions.
    
[^7]: MixMask: 重新审视Siamese ConvNets的遮盖策略

    MixMask: Revisiting Masking Strategy for Siamese ConvNets. (arXiv:2210.11456v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.11456](http://arxiv.org/abs/2210.11456)

    本文提出了一种新的填充式遮盖策略MixMask，在Siamese ConvNets中实现遮盖和对比学习目标的匹配，提高了Siamese ConvNets的性能并在多个基准测试中实现了最先进的结果。

    

    最近自监督学习的进展将Masked Image Modeling（MIM）和Siamese网络整合成一个统一的框架，利用了两种技术的优点。然而，在Siamese ConvNets中应用传统的基于擦除的遮盖策略时，存在一些未解决的问题，包括（I）在连续处理数据时不能放弃不相关的遮盖区域，导致训练效率低于ViT模型;（II）基于擦除的遮盖与Siamese ConvNets中的对比学习目标不匹配，与MIM方法不同。本文提出了一种称为MixMask的填充式遮盖策略，以防止香草遮盖方法中图像中的随机遮盖区域导致信息不完整。此外，我们引入了一种灵活的损失函数设计，考虑两个不同混合视图之间的语义距离变化，以适应集成架构并防止遮盖和对比学习目标之间的不匹配。实验表明，MixMask显着提高了Siamese ConvNets的性能，并在几个基准测试中实现了最先进的结果。

    Recent advances in self-supervised learning have integrated Masked Image Modeling (MIM) and Siamese Networks into a unified framework that leverages the benefits of both techniques. However, several issues remain unaddressed when applying conventional erase-based masking with Siamese ConvNets. These include (I) the inability to drop uninformative masked regions in ConvNets as they process data continuously, resulting in low training efficiency compared to ViT models; and (II) the mismatch between erase-based masking and the contrastive-based objective in Siamese ConvNets, which differs from the MIM approach. In this paper, we propose a filling-based masking strategy called MixMask to prevent information incompleteness caused by the randomly erased regions in an image in the vanilla masking method. Furthermore, we introduce a flexible loss function design that considers the semantic distance change between two different mixed views to adapt the integrated architecture and prevent mismat
    

