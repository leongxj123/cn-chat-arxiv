# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EC-IoU: Orienting Safety for Object Detectors via Ego-Centric Intersection-over-Union](https://arxiv.org/abs/2403.15474) | 通过EC-IoU度量，本文引入了一种定向安全性物体检测方法，可以在安全关键领域中提高物体检测器的性能，并在KITTI数据集上取得了比IoU更好的结果。 |
| [^2] | [Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models](https://arxiv.org/abs/2403.02774) | 通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。 |
| [^3] | [Neural Network Diffusion](https://arxiv.org/abs/2402.13144) | 扩散模型能够生成表现优异的神经网络参数，生成的模型在性能上与训练网络相媲美甚至更好，且成本极低。 |
| [^4] | [Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations.](http://arxiv.org/abs/2401.14142) | 基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。 |
| [^5] | [InceptionNeXt: When Inception Meets ConvNeXt.](http://arxiv.org/abs/2303.16900) | 本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。 |

# 详细

[^1]: EC-IoU: 通过自我中心交并联调整物体检测器的安全性

    EC-IoU: Orienting Safety for Object Detectors via Ego-Centric Intersection-over-Union

    [https://arxiv.org/abs/2403.15474](https://arxiv.org/abs/2403.15474)

    通过EC-IoU度量，本文引入了一种定向安全性物体检测方法，可以在安全关键领域中提高物体检测器的性能，并在KITTI数据集上取得了比IoU更好的结果。

    

    本文介绍了通过一种新颖的自我中心交并联（EC-IoU）度量来定向安全性物体检测，解决了在自动驾驶等安全关键领域应用最先进的基于学习的感知模型时面临的实际问题。具体来说，我们提出了一种加权机制来优化广泛使用的IoU度量，使其能够根据自我代理人的视角覆盖更近的地面真实对象点的预测分配更高的分数。所提出的EC-IoU度量可以用于典型的评估过程，选择有更高安全性表现的物体检测器用于下游任务。它还可以集成到常见损失函数中进行模型微调。尽管面向安全性，但我们在KITTI数据集上的实验表明，使用EC-IoU训练的模型在均值平均精度方面的性能可能会优于使用IoU训练的变体。

    arXiv:2403.15474v1 Announce Type: cross  Abstract: This paper presents safety-oriented object detection via a novel Ego-Centric Intersection-over-Union (EC-IoU) measure, addressing practical concerns when applying state-of-the-art learning-based perception models in safety-critical domains such as autonomous driving. Concretely, we propose a weighting mechanism to refine the widely used IoU measure, allowing it to assign a higher score to a prediction that covers closer points of a ground-truth object from the ego agent's perspective. The proposed EC-IoU measure can be used in typical evaluation processes to select object detectors with higher safety-related performance for downstream tasks. It can also be integrated into common loss functions for model fine-tuning. While geared towards safety, our experiment with the KITTI dataset demonstrates the performance of a model trained on EC-IoU can be better than that of a variant trained on IoU in terms of mean Average Precision as well.
    
[^2]: 快速、自适应尺度和具有不确定性意识的地球系统模型场降尺度与生成基础模型

    Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models

    [https://arxiv.org/abs/2403.02774](https://arxiv.org/abs/2403.02774)

    通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。

    

    精确和高分辨率的地球系统模型(ESM)模拟对于评估人为气候变化对生态和社会经济影响至关重要，但计算成本过高。最近的机器学习方法在ESM模拟的降尺度中表现出色，优于最先进的统计方法。然而，现有方法对每个ESM都需要计算昂贵的重新训练，并且在训练期间未见过的气候预测效果差。我们通过学习一个一致性模型(CM)，以零样本方式高效准确地降尺度任意ESM模拟来解决这些缺点。我们的基础模型方法以只受观测参考数据限制的分辨率产生概率性降尺度场。我们展示了CM在维持高可控性的同时以较低的计算成本优于最先进的扩散模型。

    arXiv:2403.02774v1 Announce Type: cross  Abstract: Accurate and high-resolution Earth system model (ESM) simulations are essential to assess the ecological and socio-economic impacts of anthropogenic climate change, but are computationally too expensive. Recent machine learning approaches have shown promising results in downscaling ESM simulations, outperforming state-of-the-art statistical approaches. However, existing methods require computationally costly retraining for each ESM and extrapolate poorly to climates unseen during training. We address these shortcomings by learning a consistency model (CM) that efficiently and accurately downscales arbitrary ESM simulations without retraining in a zero-shot manner. Our foundation model approach yields probabilistic downscaled fields at resolution only limited by the observational reference data. We show that the CM outperforms state-of-the-art diffusion models at a fraction of computational cost while maintaining high controllability on
    
[^3]: 神经网络扩散

    Neural Network Diffusion

    [https://arxiv.org/abs/2402.13144](https://arxiv.org/abs/2402.13144)

    扩散模型能够生成表现优异的神经网络参数，生成的模型在性能上与训练网络相媲美甚至更好，且成本极低。

    

    扩散模型在图像和视频生成方面取得了显著成功。在这项工作中，我们展示了扩散模型也可以\textit{生成表现优异的神经网络参数}。我们的方法很简单，利用了自动编码器和标准的潜在扩散模型。自动编码器提取了部分受训网络参数的潜在表示。然后训练了一个扩散模型来从随机噪声中合成这些潜在参数表示。它生成了新的表示，经过自动编码器的解码器，输出准备用作新的网络参数子集。在各种架构和数据集上，我们的扩散过程始终生成性能与经过训练的网络相当或更好的模型，附加成本极小。值得注意的是，我们在实证研究中发现，生成的模型与经过训练的网络表现出差异。

    arXiv:2402.13144v1 Announce Type: new  Abstract: Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also \textit{generate high-performing neural network parameters}. Our approach is simple, utilizing an autoencoder and a standard latent diffusion model. The autoencoder extracts latent representations of a subset of the trained network parameters. A diffusion model is then trained to synthesize these latent parameter representations from random noise. It then generates new representations that are passed through the autoencoder's decoder, whose outputs are ready to use as new subsets of network parameters. Across various architectures and datasets, our diffusion process consistently generates models of comparable or improved performance over trained networks, with minimal additional cost. Notably, we empirically find that the generated models perform differently with the trained networks. Our results en
    
[^4]: 基于能量的概念瓶颈模型：统一预测、概念干预和条件解释

    Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations. (arXiv:2401.14142v1 [cs.CV])

    [http://arxiv.org/abs/2401.14142](http://arxiv.org/abs/2401.14142)

    基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。

    

    现有方法，如概念瓶颈模型 (CBM)，在为黑盒深度学习模型提供基于概念的解释方面取得了成功。它们通常通过在给定输入的情况下预测概念，然后在给定预测的概念的情况下预测最终的类别标签。然而，它们经常无法捕捉到概念之间的高阶非线性相互作用，例如纠正一个预测的概念（例如“黄色胸部”）无法帮助纠正高度相关的概念（例如“黄色腹部”），导致最终准确率不理想；它们无法自然地量化不同概念和类别标签之间的复杂条件依赖关系（例如对于一个带有类别标签“Kentucky Warbler”和概念“黑色嘴巴”的图像，模型能够正确预测另一个概念“黑色冠”的概率是多少），因此无法提供关于黑盒模型工作原理更深层次的洞察。针对这些限制，我们提出了基于能量的概念瓶颈模型（Energy-based Concept Bottleneck Models）。

    Existing methods, such as concept bottleneck models (CBMs), have been successful in providing concept-based interpretations for black-box deep learning models. They typically work by predicting concepts given the input and then predicting the final class label given the predicted concepts. However, (1) they often fail to capture the high-order, nonlinear interaction between concepts, e.g., correcting a predicted concept (e.g., "yellow breast") does not help correct highly correlated concepts (e.g., "yellow belly"), leading to suboptimal final accuracy; (2) they cannot naturally quantify the complex conditional dependencies between different concepts and class labels (e.g., for an image with the class label "Kentucky Warbler" and a concept "black bill", what is the probability that the model correctly predicts another concept "black crown"), therefore failing to provide deeper insight into how a black-box model works. In response to these limitations, we propose Energy-based Concept Bot
    
[^5]: InceptionNeXt：当Inception遇到ConvNeXt

    InceptionNeXt: When Inception Meets ConvNeXt. (arXiv:2303.16900v1 [cs.CV])

    [http://arxiv.org/abs/2303.16900](http://arxiv.org/abs/2303.16900)

    本论文提出了一种名为InceptionNeXt的新型神经网络，通过将大内核卷积沿通道维度分解为四个平行分支来提高模型效率，解决了保持性能的同时加快基于大内核的CNN模型的问题。

    

    受ViTs长程建模能力的启发，近期广泛研究和采用了大内核卷积来扩大感受野和提高模型性能，例如ConvNeXt采用了7x7深度卷积。虽然这种深度操作仅消耗少量FLOPs，但由于高内存访问成本，这在功能强大的计算设备上大大损害了模型效率。尽管缩小ConvNeXt的内核大小能提高速度，但会导致性能显着下降。如何在保持性能的同时加快基于大内核的CNN模型仍不清楚。为了解决这个问题，受Inceptions的启发，我们提出将大内核深度卷积沿通道维度分解为四个平行分支，即小方内核、两个正交带内核和一个互补内核。

    Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt which employs 7x7 depthwise convolution. Although such depthwise operator only consumes a few FLOPs, it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example, ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves 60% throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed, it results in significant performance degradation. It is still unclear how to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue, inspired by Inceptions, we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension, i.e. small square kernel, two orthogonal band kernels, and an ide
    

