# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bigger is not Always Better: Scaling Properties of Latent Diffusion Models](https://arxiv.org/abs/2404.01367) | 在研究潜在扩散模型的规模特性时发现，较小的模型在相同推理预算下往往比较大的模型更有效地生成高质量结果。 |
| [^2] | [Corrupting Convolution-based Unlearnable Datasets with Pixel-based Image Transformations](https://arxiv.org/abs/2311.18403) | 研究者提出了一种基于卷积的不可学习数据集，该数据集使得现有的防御方法都失效，提出通过增加特定度量来减轻不可学习效果。 |
| [^3] | [Mitigating Feature Gap for Adversarial Robustness by Feature Disentanglement.](http://arxiv.org/abs/2401.14707) | 这项研究提出了一种通过特征解缠来缓解对抗鲁棒性中特征差距的方法，该方法明确建模和消除导致特征差距的潜在特征，有效提升了鲁棒性。 |
| [^4] | [Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection.](http://arxiv.org/abs/2308.11480) | 这项研究对机器学习中分布外检测方法进行了评估，发现现有方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不稳定。 |
| [^5] | [Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization.](http://arxiv.org/abs/2306.06805) | 提出了一种名为MACO的简单方法，通过优化相位谱生成图像，同时保持幅度恒定，解决了特征可视化在深度神经网络上的挑战，并实现了高质量和高效的可解释特征可视化。 |
| [^6] | [Score-Based Multimodal Autoencoders.](http://arxiv.org/abs/2305.15708) | 本文提出了一种基于分数模型的多模态自编码器，通过联合建模单模态VAE的潜在空间实现了对多模态数据的一致性整合，提高了多模态VAE的生成性能。 |
| [^7] | [ERM++: An Improved Baseline for Domain Generalization.](http://arxiv.org/abs/2304.01973) | ERM++是一个用于域通用性的改进基准方法，通过更好地利用训练数据、模型参数选择和权重空间正则化等关键技术，在多个数据集上比标准ERM更有效，同时计算复杂度更低，表现也优于最先进方法。 |
| [^8] | [Rethinking Alignment and Uniformity in Unsupervised Image Semantic Segmentation.](http://arxiv.org/abs/2211.14513) | 本文分析了非监督图像语义分割中困扰UISS模型的特征对齐和特征均匀性问题，提出了Semantic Attention Network(SAN) 模型，包含一个新模块 semantic attention（SEAT），以动态生成逐像素和语义特征。实验结果表明，这一非监督分割框架专注于捕捉语义表示，在多个语义分割基准测试中表现优异。 |
| [^9] | [CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning.](http://arxiv.org/abs/2210.01742) | 本文介绍了一种使用自监督对比学习进行带有对比学习的全自主分布检测的方法，能够同时检测未见过的类别和对抗性扰动样本。通过将自监督对比学习与最大均值差异（MMD）相结合，提出了CADet方法，该方法通过利用同一样本的对比变换之间的相似性进行OOD检测，并在对抗性扰动的识别方面比现有方法表现更好。 |

# 详细

[^1]: 大并非总是更好：潜在扩散模型的规模特性

    Bigger is not Always Better: Scaling Properties of Latent Diffusion Models

    [https://arxiv.org/abs/2404.01367](https://arxiv.org/abs/2404.01367)

    在研究潜在扩散模型的规模特性时发现，较小的模型在相同推理预算下往往比较大的模型更有效地生成高质量结果。

    

    我们研究了潜在扩散模型（LDMs）的规模特性，重点关注它们的采样效率。尽管改进的网络架构和推理算法已经证明可以有效提升扩散模型的采样效率，但模型大小的作用——采样效率的关键决定因素——尚未受到彻底的审查。通过对已建立的文本到图像扩散模型的实证分析，我们进行了深入研究，探讨了模型大小如何影响在不同采样步骤下的采样效率。我们的发现揭示了一个令人惊讶的趋势：在给定推理预算下运行时，较小的模型经常胜过其较大的等价物在生成高质量结果上。此外，我们扩展了研究，通过应用各种扩散采样器，探索不同的下游任务，评估后精馏模型，以及进行比较，来展示这些发现的普适性。

    arXiv:2404.01367v1 Announce Type: cross  Abstract: We study the scaling properties of latent diffusion models (LDMs) with an emphasis on their sampling efficiency. While improved network architecture and inference algorithms have shown to effectively boost sampling efficiency of diffusion models, the role of model size -- a critical determinant of sampling efficiency -- has not been thoroughly examined. Through empirical analysis of established text-to-image diffusion models, we conduct an in-depth investigation into how model size influences sampling efficiency across varying sampling steps. Our findings unveil a surprising trend: when operating under a given inference budget, smaller models frequently outperform their larger equivalents in generating high-quality results. Moreover, we extend our study to demonstrate the generalizability of the these findings by applying various diffusion samplers, exploring diverse downstream tasks, evaluating post-distilled models, as well as compar
    
[^2]: 使用基于像素的图像转换破坏基于卷积的不可学习数据集

    Corrupting Convolution-based Unlearnable Datasets with Pixel-based Image Transformations

    [https://arxiv.org/abs/2311.18403](https://arxiv.org/abs/2311.18403)

    研究者提出了一种基于卷积的不可学习数据集，该数据集使得现有的防御方法都失效，提出通过增加特定度量来减轻不可学习效果。

    

    不可学习的数据集会通过向干净训练集引入精心设计且难以察觉的扰动，导致模型的泛化性能急剧下降。许多现有防御方法，如JPEG压缩和对抗训练，能够有效对抗基于范数约束的附加噪声的不可学习数据集。然而，最新提出的一种基于卷积的不可学习数据集让现有的防御方法无效，给防御者带来更大挑战。为了解决这个问题，我们在简化的情景中将基于卷积的不可学习样本表达为将矩阵乘以干净样本的结果，并将类内矩阵不一致性形式化为$\Theta_{imi}$，将类间矩阵一致性形式化为$\Theta_{imc}$以研究基于卷积的不可学习数据集的工作机制。我们推测增加这两个度量将有助于减轻不可学习效果。

    arXiv:2311.18403v2 Announce Type: replace-cross  Abstract: Unlearnable datasets lead to a drastic drop in the generalization performance of models trained on them by introducing elaborate and imperceptible perturbations into clean training sets. Many existing defenses, e.g., JPEG compression and adversarial training, effectively counter UDs based on norm-constrained additive noise. However, a fire-new type of convolution-based UDs have been proposed and render existing defenses all ineffective, presenting a greater challenge to defenders. To address this, we express the convolution-based unlearnable sample as the result of multiplying a matrix by a clean sample in a simplified scenario, and formalize the intra-class matrix inconsistency as $\Theta_{imi}$, inter-class matrix consistency as $\Theta_{imc}$ to investigate the working mechanism of the convolution-based UDs. We conjecture that increasing both of these metrics will mitigate the unlearnability effect. Through validation experi
    
[^3]: 通过特征解缠来缓解对抗鲁棒性中的特征差距

    Mitigating Feature Gap for Adversarial Robustness by Feature Disentanglement. (arXiv:2401.14707v1 [cs.CV])

    [http://arxiv.org/abs/2401.14707](http://arxiv.org/abs/2401.14707)

    这项研究提出了一种通过特征解缠来缓解对抗鲁棒性中特征差距的方法，该方法明确建模和消除导致特征差距的潜在特征，有效提升了鲁棒性。

    

    深度神经网络对对抗样本很容易受到攻击。对抗微调方法旨在通过对已经在自然情况下进行预训练的模型进行对抗式微调来提升对抗鲁棒性。然而，我们发现对抗样本中的一些潜在特征被对抗扰动所混淆，并导致自然样本和对抗样本在最后一层隐藏层的特征之间出现意外增加的差距。为了解决这个问题，我们提出了一种基于解缠的方法来明确建模和进一步消除导致特征差距的潜在特征。具体而言，我们引入了特征解缠器，将对抗样本的潜在特征与对抗样本的特征分离开来，从而通过消除潜在特征来提升鲁棒性。此外，我们通过将预训练模型中的特征与对抗样本在微调模型中的特征对齐，进一步从自然样本的特征中获益，避免混淆。

    Deep neural networks are vulnerable to adversarial samples. Adversarial fine-tuning methods aim to enhance adversarial robustness through fine-tuning the naturally pre-trained model in an adversarial training manner. However, we identify that some latent features of adversarial samples are confused by adversarial perturbation and lead to an unexpectedly increasing gap between features in the last hidden layer of natural and adversarial samples. To address this issue, we propose a disentanglement-based approach to explicitly model and further remove the latent features that cause the feature gap. Specifically, we introduce a feature disentangler to separate out the latent features from the features of the adversarial samples, thereby boosting robustness by eliminating the latent features. Besides, we align features in the pre-trained model with features of adversarial samples in the fine-tuned model, to further benefit from the features from natural samples without confusion. Empirical 
    
[^4]: 对广泛的分布外检测的期望：期望之外的未知数据

    Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection. (arXiv:2308.11480v1 [cs.LG])

    [http://arxiv.org/abs/2308.11480](http://arxiv.org/abs/2308.11480)

    这项研究对机器学习中分布外检测方法进行了评估，发现现有方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不稳定。

    

    提高部署的机器学习系统的可靠性通常涉及开发方法来检测分布外（OOD）的输入。然而，现有研究常常狭窄地关注训练集中缺失的类别样本，忽略了其他类型的可能分布变化。这种限制降低了这些方法在现实场景中的适用性，因为系统会遇到各种各样的异常输入。在本研究中，我们将五种不同类型的分布变化进行分类，并对最近的OOD检测方法在每一种分布变化上进行了关键评估。我们以BROAD（Benchmarking Resilience Over Anomaly Diversity）的名义公开发布我们的基准。我们的研究发现这些方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不一致。换句话说，它们只能可靠地检测到它们特别设计来预期的意外输入。

    Improving the reliability of deployed machine learning systems often involves developing methods to detect out-of-distribution (OOD) inputs. However, existing research often narrowly focuses on samples from classes that are absent from the training set, neglecting other types of plausible distribution shifts. This limitation reduces the applicability of these methods in real-world scenarios, where systems encounter a wide variety of anomalous inputs. In this study, we categorize five distinct types of distribution shifts and critically evaluate the performance of recent OOD detection methods on each of them. We publicly release our benchmark under the name BROAD (Benchmarking Resilience Over Anomaly Diversity). Our findings reveal that while these methods excel in detecting unknown classes, their performance is inconsistent when encountering other types of distribution shifts. In other words, they only reliably detect unexpected inputs that they have been specifically designed to expec
    
[^5]: 用幅度受限制优化解锁更深层网络的特征可视化

    Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization. (arXiv:2306.06805v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.06805](http://arxiv.org/abs/2306.06805)

    提出了一种名为MACO的简单方法，通过优化相位谱生成图像，同时保持幅度恒定，解决了特征可视化在深度神经网络上的挑战，并实现了高质量和高效的可解释特征可视化。

    

    特征可视化在Olah等人2017年的有影响力的工作之后获得了很大的 popularity，将其确立为可解释性的重要工具。然而，由于依赖于生成可解释图像的技巧以及在将其扩展到更深的神经网络时面临的挑战，其广泛应用受到了限制。在这里，我们描述了一种简单的方法MACO来解决这些问题。主要思想是通过优化相位谱生成图像，同时保持幅度恒定，以确保生成的解释位于自然图像空间中。我们的方法在定性和定量上都取得了显着的改进，并为大型最先进神经网络提供了高效且可解释的特征可视化。我们还展示了我们的方法具有一个属性机制，可以增强特征可视化的空间重要性。我们在一个新的基准测试中验证了我们的方法。

    Feature visualization has gained substantial popularity, particularly after the influential work by Olah et al. in 2017, which established it as a crucial tool for explainability. However, its widespread adoption has been limited due to a reliance on tricks to generate interpretable images, and corresponding challenges in scaling it to deeper neural networks. Here, we describe MACO, a simple approach to address these shortcomings. The main idea is to generate images by optimizing the phase spectrum while keeping the magnitude constant to ensure that generated explanations lie in the space of natural images. Our approach yields significantly better results (both qualitatively and quantitatively) and unlocks efficient and interpretable feature visualizations for large state-of-the-art neural networks. We also show that our approach exhibits an attribution mechanism allowing us to augment feature visualizations with spatial importance. We validate our method on a novel benchmark for compa
    
[^6]: 基于分数的多模态自编码器

    Score-Based Multimodal Autoencoders. (arXiv:2305.15708v1 [cs.LG])

    [http://arxiv.org/abs/2305.15708](http://arxiv.org/abs/2305.15708)

    本文提出了一种基于分数模型的多模态自编码器，通过联合建模单模态VAE的潜在空间实现了对多模态数据的一致性整合，提高了多模态VAE的生成性能。

    

    多模态变分自编码器是一类能够在潜在空间中构建可处理后验的有前途的生成模型，特别适用于多种模态的数据。但随着模态数量的增加，每一个模态的生成质量都会降低。本文提出了一种新的方法，通过使用基于分数的模型联合建模单模态VAE的潜在空间，以增强多模态VAE的生成性能。分数模型的作用是通过学习潜在变量之间的相关性来实现多模态的一致性。因此，我们的模型结合了单模态VAE卓越的生成质量和对不同模态的一致性整合。

    Multimodal Variational Autoencoders (VAEs) represent a promising group of generative models that facilitate the construction of a tractable posterior within the latent space, given multiple modalities. Daunhawer et al. (2022) demonstrate that as the number of modalities increases, the generative quality of each modality declines. In this study, we explore an alternative approach to enhance the generative performance of multimodal VAEs by jointly modeling the latent space of unimodal VAEs using score-based models (SBMs). The role of the SBM is to enforce multimodal coherence by learning the correlation among the latent variables. Consequently, our model combines the superior generative quality of unimodal VAEs with coherent integration across different modalities.
    
[^7]: ERM++：用于域通用性的改进基准方法

    ERM++: An Improved Baseline for Domain Generalization. (arXiv:2304.01973v1 [cs.LG])

    [http://arxiv.org/abs/2304.01973](http://arxiv.org/abs/2304.01973)

    ERM++是一个用于域通用性的改进基准方法，通过更好地利用训练数据、模型参数选择和权重空间正则化等关键技术，在多个数据集上比标准ERM更有效，同时计算复杂度更低，表现也优于最先进方法。

    

    多源域通用性（DG）衡量分类器对于它没有接受过训练的新数据分布的泛化能力，并考虑了多个训练域。虽然已经提出了几种多源DG方法，但是它们在训练过程中使用域标签增加了额外的复杂性。最近的研究表明，经过良好调整的经验风险最小化（ERM）训练过程，即在源域上简单地最小化经验风险，可以胜过大多数现有的DG方法。我们确定了几个关键候选技术，以进一步提高ERM的性能，例如更好地利用训练数据、模型参数选择和权重空间正则化。我们将结果称为ERM ++，并展示它相对于标准ERM在五个多源数据集上将DG的性能显着提高了5％以上，并且尽管计算复杂度更低，但击败了最先进的方法。此外，我们还证明了ERM ++在WILDS-FMOW数据集上的有效性。

    Multi-source Domain Generalization (DG) measures a classifier's ability to generalize to new distributions of data it was not trained on, given several training domains. While several multi-source DG methods have been proposed, they incur additional complexity during training by using domain labels. Recent work has shown that a well-tuned Empirical Risk Minimization (ERM) training procedure, that is simply minimizing the empirical risk on the source domains, can outperform most existing DG methods. We identify several key candidate techniques to further improve ERM performance, such as better utilization of training data, model parameter selection, and weight-space regularization. We call the resulting method ERM++, and show it significantly improves the performance of DG on five multi-source datasets by over 5% compared to standard ERM, and beats state-of-the-art despite being less computationally expensive. Additionally, we demonstrate the efficacy of ERM++ on the WILDS-FMOW dataset,
    
[^8]: 重新思考非监督图像语义分割中的对齐和均匀性问题

    Rethinking Alignment and Uniformity in Unsupervised Image Semantic Segmentation. (arXiv:2211.14513v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.14513](http://arxiv.org/abs/2211.14513)

    本文分析了非监督图像语义分割中困扰UISS模型的特征对齐和特征均匀性问题，提出了Semantic Attention Network(SAN) 模型，包含一个新模块 semantic attention（SEAT），以动态生成逐像素和语义特征。实验结果表明，这一非监督分割框架专注于捕捉语义表示，在多个语义分割基准测试中表现优异。

    

    非监督图像语义分割(UISS)旨在将低层视觉特征与语义级别的表示匹配，而无需外部监管。本文从特征对齐和特征均匀性的角度探究了UISS模型的关键性质，并将UISS与整幅图像的表示学习进行了比较。基于分析，我们认为UISS中现有的基于互信息的方法存在表示崩溃的问题。因此，我们提出了一种稳健的网络模型——Semantic Attention Network(SAN)，其中提出了一种新模块Semantic Attention(SEAT)，以动态生成逐像素和语义特征。在多个语义分割基准测试中的实验结果表明，我们的非监督分割框架专注于捕捉语义表示，表现优异，超过了所有未预训练的模型，甚至超过了一些预训练模型。

    Unsupervised image semantic segmentation(UISS) aims to match low-level visual features with semantic-level representations without outer supervision. In this paper, we address the critical properties from the view of feature alignments and feature uniformity for UISS models. We also make a comparison between UISS and image-wise representation learning. Based on the analysis, we argue that the existing MI-based methods in UISS suffer from representation collapse. By this, we proposed a robust network called Semantic Attention Network(SAN), in which a new module Semantic Attention(SEAT) is proposed to generate pixel-wise and semantic features dynamically. Experimental results on multiple semantic segmentation benchmarks show that our unsupervised segmentation framework specializes in catching semantic representations, which outperforms all the unpretrained and even several pretrained methods.
    
[^9]: CADet:全自监督对比学习用于带有对比学习的全自主分布检测

    CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning. (arXiv:2210.01742v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.01742](http://arxiv.org/abs/2210.01742)

    本文介绍了一种使用自监督对比学习进行带有对比学习的全自主分布检测的方法，能够同时检测未见过的类别和对抗性扰动样本。通过将自监督对比学习与最大均值差异（MMD）相结合，提出了CADet方法，该方法通过利用同一样本的对比变换之间的相似性进行OOD检测，并在对抗性扰动的识别方面比现有方法表现更好。

    

    处理带有分布之外（OOD）样本已成为机器学习系统在现实世界部署中的主要问题。本文探讨了使用自监督对比学习同时检测两种类型的OOD样本：未见过的类别和对抗性扰动。首先，我们将自监督对比学习与最大均值差异（MMD）双样本检验相结合。这种方法使我们能够鲁棒地测试两个独立样本集是否来自相同的分布，并且我们证明了相较于之前的工作，它在区分CIFAR-10和CIFAR-10.1时具有更高的置信度。在此成功的基础上，我们引入了CADet（对比异常检测），一种用于单样本OOD检测的新方法。CADet借鉴了MMD的思想，但利用了同一样本的对比变换之间的相似性。CADet在识别被对抗性扰动干扰的样本方面胜过现有的对抗检测方法。

    Handling out-of-distribution (OOD) samples has become a major stake in the real-world deployment of machine learning systems. This work explores the use of self-supervised contrastive learning to the simultaneous detection of two types of OOD samples: unseen classes and adversarial perturbations. First, we pair self-supervised contrastive learning with the maximum mean discrepancy (MMD) two-sample test. This approach enables us to robustly test whether two independent sets of samples originate from the same distribution, and we demonstrate its effectiveness by discriminating between CIFAR-10 and CIFAR-10.1 with higher confidence than previous work. Motivated by this success, we introduce CADet (Contrastive Anomaly Detection), a novel method for OOD detection of single samples. CADet draws inspiration from MMD, but leverages the similarity between contrastive transformations of a same sample. CADet outperforms existing adversarial detection methods in identifying adversarially perturbed
    

