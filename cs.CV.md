# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [STARFlow: Spatial Temporal Feature Re-embedding with Attentive Learning for Real-world Scene Flow](https://arxiv.org/abs/2403.07032) | 提出了一种全局注意力流嵌入和空间时间特征重新嵌入模块相结合的方法，用于解决现实世界场景流预测中的局部依赖匹配和非刚性物体变形的挑战。 |
| [^2] | [IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images](https://arxiv.org/abs/2402.03227) | IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。 |
| [^3] | [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417) | 本文提出了一种新的通用视觉骨干模型Vim，通过使用双向状态空间模型和位置嵌入来高效表示视觉数据，相比于传统的视觉转换器如DeiT，在各种视觉任务上取得了更好的性能，并且实现了显著提升。 |
| [^4] | [The cell signaling structure function.](http://arxiv.org/abs/2401.02501) | 该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。 |

# 详细

[^1]: STARFlow: 具有注意力学习的空间时间特征重新嵌入用于现实世界的场景流

    STARFlow: Spatial Temporal Feature Re-embedding with Attentive Learning for Real-world Scene Flow

    [https://arxiv.org/abs/2403.07032](https://arxiv.org/abs/2403.07032)

    提出了一种全局注意力流嵌入和空间时间特征重新嵌入模块相结合的方法，用于解决现实世界场景流预测中的局部依赖匹配和非刚性物体变形的挑战。

    

    场景流预测是理解动态场景中的关键任务，因为它提供了基本的运动信息。然而，当代场景流方法面临三大挑战。首先，仅基于局部感受野的流估计缺乏点对的长依赖匹配。为了解决这个问题，我们提出了全局注意力流嵌入，以匹配特征空间和欧几里得空间中的所有点对，提供局部细化之前的全局初始化。其次，在变形后存在非刚性物体的变形，导致连续帧之间的时空关系变化。为了更精确地估计残余流，设计了一个空间时间特征重新嵌入模块，以在变形后获取序列特征。此外，由于合成数据和真实数据之间的显著域差异，先前的方法表现出较差的泛化能力。

    arXiv:2403.07032v1 Announce Type: cross  Abstract: Scene flow prediction is a crucial underlying task in understanding dynamic scenes as it offers fundamental motion information. However, contemporary scene flow methods encounter three major challenges. Firstly, flow estimation solely based on local receptive fields lacks long-dependency matching of point pairs. To address this issue, we propose global attentive flow embedding to match all-to-all point pairs in both feature space and Euclidean space, providing global initialization before local refinement. Secondly, there are deformations existing in non-rigid objects after warping, which leads to variations in the spatiotemporal relation between the consecutive frames. For a more precise estimation of residual flow, a spatial temporal feature re-embedding module is devised to acquire the sequence features after deformation. Furthermore, previous methods perform poor generalization due to the significant domain gap between the synthesi
    
[^2]: IGUANe: 一种适用于脑MR图像多中心协调的三维通用CycleGAN模型

    IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images

    [https://arxiv.org/abs/2402.03227](https://arxiv.org/abs/2402.03227)

    IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。

    

    在MRI研究中，来自多个采集点的图像数据的聚合可以增加样本大小，但可能引入阻碍后续分析一致性的与采集点相关的变异。图像翻译的深度学习方法已经成为协调MR图像跨站点的解决方案。在本研究中，我们引入了IGUANe（具有统一对抗网络的图像生成），这是一种原始的三维模型，它结合了域转换的优势和直接应用样式转移方法来实现多中心脑MR图像协调。IGUANe通过多对一策略，集成了任意数量的域进行训练，扩展了CycleGAN架构。在推断过程中，该模型可以应用于任何图像，甚至来自未知采集点，使其成为协调的通用生成器。在由11台不同扫描仪的T1加权图像组成的数据集上进行训练，IGUANe在未见站点的数据上进行了评估。

    In MRI studies, the aggregation of imaging data from multiple acquisition sites enhances sample size but may introduce site-related variabilities that hinder consistency in subsequent analyses. Deep learning methods for image translation have emerged as a solution for harmonizing MR images across sites. In this study, we introduce IGUANe (Image Generation with Unified Adversarial Networks), an original 3D model that leverages the strengths of domain translation and straightforward application of style transfer methods for multicenter brain MR image harmonization. IGUANe extends CycleGAN architecture by integrating an arbitrary number of domains for training through a many-to-one strategy. During inference, the model can be applied to any image, even from an unknown acquisition site, making it a universal generator for harmonization. Trained on a dataset comprising T1-weighted images from 11 different scanners, IGUANe was evaluated on data from unseen sites. The assessments included the
    
[^3]: Vision Mamba: 使用双向状态空间模型高效学习视觉表示

    Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

    [https://arxiv.org/abs/2401.09417](https://arxiv.org/abs/2401.09417)

    本文提出了一种新的通用视觉骨干模型Vim，通过使用双向状态空间模型和位置嵌入来高效表示视觉数据，相比于传统的视觉转换器如DeiT，在各种视觉任务上取得了更好的性能，并且实现了显著提升。

    

    最近，具有高效硬件感知设计的状态空间模型（SSMs），即Mamba深度学习模型，在长序列建模方面展现出巨大潜力。与此同时，在SSMs上构建高效且通用的视觉骨干模型也是一种有吸引力的方向。然而，由于视觉数据的位置敏感性和对全局上下文的需求，对于SSMs来说，表示视觉数据是具有挑战性的。在本文中，我们展示了对于视觉表示学习来说，依赖自注意力并不是必要的，并提出了一种新的通用视觉骨干模型，即带有双向Mamba块（Vim），它使用位置嵌入标记图像序列，并使用双向状态空间模型压缩视觉表示。在ImageNet分类、COCO目标检测和ADE20k语义分割任务中，Vim相比于DeiT等经过良好验证的视觉转换器，实现了更高的性能，并且显示出了显著的改进。

    Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., the Mamba deep learning model, have shown great potential for long sequence modeling. Meanwhile building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance on self-attention for visual representation learning is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved
    
[^4]: 细胞信号传导结构和功能

    The cell signaling structure function. (arXiv:2401.02501v1 [cs.CV])

    [http://arxiv.org/abs/2401.02501](http://arxiv.org/abs/2401.02501)

    该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。

    

    活体细胞显微镜捕捉到的五维$(x,y,z,channel,time)$视频显示了细胞运动和信号动力学的模式。我们在这里提出一种在五维活体细胞显微镜视频中寻找细胞信号动力学时空模式的方法，该方法独特之处在于不需要预先了解预期的模式动力学以及没有训练数据。所提出的细胞信号结构函数（SSF）是一种Kolmogorov结构函数，可以通过核心区域相对于周围细胞质的核糖体强度来最优地测量细胞信号状态，相比当前最先进的核糖体与细胞核比值有了显著的改进。通过度量归一化压缩距离（NCD）来识别相似的模式。NCD是一个用于表示输入的SSF构图在低维嵌入中作为点的Hilbert空间的再生核，可以最优地捕捉模式。

    Live cell microscopy captures 5-D $(x,y,z,channel,time)$ movies that display patterns of cellular motion and signaling dynamics. We present here an approach to finding spatiotemporal patterns of cell signaling dynamics in 5-D live cell microscopy movies unique in requiring no \emph{a priori} knowledge of expected pattern dynamics, and no training data. The proposed cell signaling structure function (SSF) is a Kolmogorov structure function that optimally measures cell signaling state as nuclear intensity w.r.t. surrounding cytoplasm, a significant improvement compared to the current state-of-the-art cytonuclear ratio. SSF kymographs store at each spatiotemporal cell centroid the SSF value, or a functional output such as velocity. Patterns of similarity are identified via the metric normalized compression distance (NCD). The NCD is a reproducing kernel for a Hilbert space that represents the input SSF kymographs as points in a low dimensional embedding that optimally captures the pattern
    

