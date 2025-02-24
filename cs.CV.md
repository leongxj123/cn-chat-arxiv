# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://arxiv.org/abs/2403.17465) | LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。 |
| [^2] | [MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations](https://arxiv.org/abs/2402.10093) | MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。 |
| [^3] | [Towards Robust Probabilistic Modeling on SO(3) via Rotation Laplace Distribution.](http://arxiv.org/abs/2305.10465) | 本文提出了一种新的基于旋转拉普拉斯分布的SO(3)稳健概率建模方法，对异常值具有鲁棒性，并可以容忍不完美的注释。 |

# 详细

[^1]: LaRE^2: 基于潜在重构误差的扩散生成图像检测方法

    LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection

    [https://arxiv.org/abs/2403.17465](https://arxiv.org/abs/2403.17465)

    LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。

    

    arXiv:2403.17465v1 类型：交叉 摘要：扩散模型的发展显著提高了图像生成质量，使真实图像和生成图像之间的区分变得越来越困难。尽管这一进展令人印象深刻，但也引发了重要的隐私和安全问题。为了解决这一问题，我们提出了一种新颖的基于潜在重构误差引导特征细化方法（LaRE^2）来检测扩散生成的图像。我们提出了潜在重构误差（LaRE），作为潜在空间中生成图像检测的第一个基于重构误差的特征。LaRE在特征提取效率方面超越了现有方法，同时保留了区分真假所需的关键线索。为了利用LaRE，我们提出了一种误差引导特征细化模块（EGRE），它可以通过LaRE引导的方式细化图像特征，以增强特征的区分能力。

    arXiv:2403.17465v1 Announce Type: cross  Abstract: The evolution of Diffusion Models has dramatically improved image generation quality, making it increasingly difficult to differentiate between real and generated images. This development, while impressive, also raises significant privacy and security concerns. In response to this, we propose a novel Latent REconstruction error guided feature REfinement method (LaRE^2) for detecting the diffusion-generated images. We come up with the Latent Reconstruction Error (LaRE), the first reconstruction-error based feature in the latent space for generated image detection. LaRE surpasses existing methods in terms of feature extraction efficiency while preserving crucial cues required to differentiate between the real and the fake. To exploit LaRE, we propose an Error-Guided feature REfinement module (EGRE), which can refine the image feature guided by LaRE to enhance the discriminativeness of the feature. Our EGRE utilizes an align-then-refine m
    
[^2]: MIM-Refiner：一种从中间预训练表示中获得对比学习提升的方法

    MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

    [https://arxiv.org/abs/2402.10093](https://arxiv.org/abs/2402.10093)

    MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。

    

    我们引入了MIM-Refiner，这是一种用于预训练MIM模型的对比学习提升方法。MIM-Refiner的动机在于MIM模型中的最佳表示通常位于中间层。因此，MIM-Refiner利用连接到不同中间层的多个对比头。在每个头中，修改后的最近邻目标帮助构建相应的语义聚类。此过程短而有效，在几个epochs内，我们将MIM模型的特征从次优的状态提升到最先进的状态。使用data2vec 2.0在ImageNet-1K上预训练的ViT-H经过改进后，在线性探测和低样本分类方面取得了新的最先进结果（分别为84.7%和64.2%），超过了在ImageNet-1K上预训练的其他模型的表现。

    arXiv:2402.10093v1 Announce Type: cross  Abstract: We introduce MIM (Masked Image Modeling)-Refiner, a contrastive learning boost for pre-trained MIM models. The motivation behind MIM-Refiner is rooted in the insight that optimal representations within MIM models generally reside in intermediate layers. Accordingly, MIM-Refiner leverages multiple contrastive heads that are connected to diverse intermediate layers. In each head, a modified nearest neighbor objective helps to construct respective semantic clusters.   The refinement process is short but effective. Within a few epochs, we refine the features of MIM models from subpar to state-of-the-art, off-the-shelf features. Refining a ViT-H, pre-trained with data2vec 2.0 on ImageNet-1K, achieves new state-of-the-art results in linear probing (84.7%) and low-shot classification among models that are pre-trained on ImageNet-1K. In ImageNet-1K 1-shot classification, MIM-Refiner sets a new state-of-the-art of 64.2%, outperforming larger mo
    
[^3]: 基于旋转拉普拉斯分布的SO(3)稳健概率建模研究

    Towards Robust Probabilistic Modeling on SO(3) via Rotation Laplace Distribution. (arXiv:2305.10465v1 [cs.CV])

    [http://arxiv.org/abs/2305.10465](http://arxiv.org/abs/2305.10465)

    本文提出了一种新的基于旋转拉普拉斯分布的SO(3)稳健概率建模方法，对异常值具有鲁棒性，并可以容忍不完美的注释。

    

    从单张RGB图像估计三维自由旋转是一项重要且具有挑战性的任务。概率旋转建模是一种流行的方法，相对于单预测旋转回归可以额外提供预测不确定性信息。对于SO(3)上的概率分布建模，使用类似于高斯的Bingham分布和矩阵Fisher分布是自然的，但是它们对异常预测很敏感，例如180度误差，因此不太可能以最佳性能收敛。本文从多元拉普拉斯分布中汲取灵感，提出了一种新的SO(3)旋转拉普拉斯分布。我们的旋转拉普拉斯分布对异常值的干扰具有鲁棒性，并强制施加梯度到低误差区域，以改进性能。此外，我们还证明了我们的方法对小噪声具有鲁棒性，因此可以容忍不完美的注释。利用这个优势，我们展示了在半监督回归任务上的优势。

    Estimating the 3DoF rotation from a single RGB image is an important yet challenging problem. As a popular approach, probabilistic rotation modeling additionally carries prediction uncertainty information, compared to single-prediction rotation regression. For modeling probabilistic distribution over SO(3), it is natural to use Gaussian-like Bingham distribution and matrix Fisher, however they are shown to be sensitive to outlier predictions, e.g. $180^\circ$ error and thus are unlikely to converge with optimal performance. In this paper, we draw inspiration from multivariate Laplace distribution and propose a novel rotation Laplace distribution on SO(3). Our rotation Laplace distribution is robust to the disturbance of outliers and enforces much gradient to the low-error region that it can improve. In addition, we show that our method also exhibits robustness to small noises and thus tolerates imperfect annotations. With this benefit, we demonstrate its advantages in semi-supervised r
    

