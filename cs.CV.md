# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unconditional Latent Diffusion Models Memorize Patient Imaging Data](https://rss.arxiv.org/abs/2402.01054) | 本论文研究了医学图像合成中隐式扩散模型的记忆问题。通过评估训练数据的记忆程度以及探索可能导致记忆的因素，揭示了这一问题的重要性和潜在风险。 |
| [^2] | [Conjugate-Gradient-like Based Adaptive Moment Estimation Optimization Algorithm for Deep Learning](https://arxiv.org/abs/2404.01714) | 提出一种基于共轭梯度样式的新优化算法CG-like-Adam，用于深度学习，并在收敛分析和数值实验中展示了其优越性 |
| [^3] | [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103) | 该教程讨论了图像和视觉领域中扩散模型的基本理念，适合对扩散模型研究或应用感兴趣的本科生和研究生。 |
| [^4] | [NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion](https://arxiv.org/abs/2402.13809) | NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。 |
| [^5] | [DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision.](http://arxiv.org/abs/2309.06941) | 该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。 |
| [^6] | [Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI.](http://arxiv.org/abs/2308.05764) | 该论文提出了一种通过从心脏MRI中的知识转移解锁心电图的诊断潜力的方法。通过将CMR图像中的领域特定信息转移到ECG嵌入中，该方法实现了仅根据ECG数据进行全面的心脏筛查，并能预测心血管疾病的个体风险和确定心脏表型。 |

# 详细

[^1]: 无条件的隐式扩散模型记忆患者影像数据

    Unconditional Latent Diffusion Models Memorize Patient Imaging Data

    [https://rss.arxiv.org/abs/2402.01054](https://rss.arxiv.org/abs/2402.01054)

    本论文研究了医学图像合成中隐式扩散模型的记忆问题。通过评估训练数据的记忆程度以及探索可能导致记忆的因素，揭示了这一问题的重要性和潜在风险。

    

    生成式的隐式扩散模型在医学影像领域具有广泛的应用。一个值得注意的应用是通过提出合成数据作为真实患者数据的替代品来实现隐私保护的开放数据共享。尽管有这个应用的前景，但这些模型容易出现患者数据的记忆问题，即模型生成患者数据的副本而不是新的合成样本。这破坏了保护患者数据的整个目的，甚至可能导致患者被重新识别。考虑到这个问题的重要性，令人惊讶的是，在医学影像界中这个问题并没有得到太多关注。为此，我们评估了医学图像合成中隐式扩散模型的记忆问题。我们训练了2D和3D的隐式扩散模型，使用CT、MR和X光数据集进行合成数据的生成。之后，我们利用自监督模型来评估训练数据被记忆的程度，并进一步研究可能导致记忆的各种因素。

    Generative latent diffusion models hold a wide range of applications in the medical imaging domain. A noteworthy application is privacy-preserved open-data sharing by proposing synthetic data as surrogates of real patient data. Despite the promise, these models are susceptible to patient data memorization, where models generate patient data copies instead of novel synthetic samples. This undermines the whole purpose of preserving patient data and may even result in patient re-identification. Considering the importance of the problem, surprisingly it has received relatively little attention in the medical imaging community. To this end, we assess memorization in latent diffusion models for medical image synthesis. We train 2D and 3D latent diffusion models on CT, MR, and X-ray datasets for synthetic data generation. Afterwards, we examine the amount of training data memorized utilizing self-supervised models and further investigate various factors that can possibly lead to memorization 
    
[^2]: 基于共轭梯度的自适应矩估计优化算法用于深度学习

    Conjugate-Gradient-like Based Adaptive Moment Estimation Optimization Algorithm for Deep Learning

    [https://arxiv.org/abs/2404.01714](https://arxiv.org/abs/2404.01714)

    提出一种基于共轭梯度样式的新优化算法CG-like-Adam，用于深度学习，并在收敛分析和数值实验中展示了其优越性

    

    训练深度神经网络是一项具有挑战性的任务。为加快培训速度并增强深度神经网络的性能，我们将传统的共轭梯度修正为共轭梯度样式，并将其并入通用Adam中，因此提出了一种名为CG-like-Adam的新优化算法用于深度学习。具体而言，通用Adam的一阶和二阶矩估计均由共轭梯度样式替换。收敛分析处理了一阶矩估计的指数移动平均系数为常数且一阶矩估计无偏的情况。数值实验显示了基于CIFAR10/100数据集的所提算法的优越性。

    arXiv:2404.01714v1 Announce Type: cross  Abstract: Training deep neural networks is a challenging task. In order to speed up training and enhance the performance of deep neural networks, we rectify the vanilla conjugate gradient as conjugate-gradient-like and incorporate it into the generic Adam, and thus propose a new optimization algorithm named CG-like-Adam for deep learning. Specifically, both the first-order and the second-order moment estimation of generic Adam are replaced by the conjugate-gradient-like. Convergence analysis handles the cases where the exponential moving average coefficient of the first-order moment estimation is constant and the first-order moment estimation is unbiased. Numerical experiments show the superiority of the proposed algorithm based on the CIFAR10/100 dataset.
    
[^3]: 关于图像和视觉扩散模型的教程

    Tutorial on Diffusion Models for Imaging and Vision

    [https://arxiv.org/abs/2403.18103](https://arxiv.org/abs/2403.18103)

    该教程讨论了图像和视觉领域中扩散模型的基本理念，适合对扩散模型研究或应用感兴趣的本科生和研究生。

    

    近年来生成工具的惊人增长使得文本到图像生成和文本到视频生成等许多令人兴奋的应用成为可能。这些生成工具背后的基本原理是扩散概念，一种特殊的采样机制，克服了以前方法中被认为困难的一些缺点。本教程的目标是讨论扩散模型的基本理念。本教程的目标受众包括对研究扩散模型或将这些模型应用于解决其他问题感兴趣的本科生和研究生。

    arXiv:2403.18103v1 Announce Type: new  Abstract: The astonishing growth of generative tools in recent years has empowered many exciting applications in text-to-image generation and text-to-video generation. The underlying principle behind these generative tools is the concept of diffusion, a particular sampling mechanism that has overcome some shortcomings that were deemed difficult in the previous approaches. The goal of this tutorial is to discuss the essential ideas underlying the diffusion models. The target audience of this tutorial includes undergraduate and graduate students who are interested in doing research on diffusion models or applying these models to solve other problems.
    
[^4]: NeuralDiffuser：具有主视觉特征引导扩散的可控fMRI重建

    NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion

    [https://arxiv.org/abs/2402.13809](https://arxiv.org/abs/2402.13809)

    NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。

    

    基于潜在扩散模型(LDM)从功能性磁共振成像(fMRI)中重建视觉刺激，为大脑提供了细粒度的检索。一个挑战在于重建细节的连贯对齐（如结构、背景、纹理、颜色等）。此外，即使在相同条件下，LDM也会生成不同的图像结果。因此，我们首先揭示了基于LDM的神经科学视角，即基于来自海量图像的预训练知识进行自上而下的创建，但缺乏基于细节驱动的自下而上感知，导致细节不忠实。我们提出了NeuralDiffuser，引入主视觉特征引导，以渐变形式提供细节线索，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。我们还开发了一种新颖的引导策略，以确保重复重建的一致性，而不是随机性。

    arXiv:2402.13809v1 Announce Type: cross  Abstract: Reconstructing visual stimuli from functional Magnetic Resonance Imaging (fMRI) based on Latent Diffusion Models (LDM) provides a fine-grained retrieval of the brain. A challenge persists in reconstructing a cohesive alignment of details (such as structure, background, texture, color, etc.). Moreover, LDMs would generate different image results even under the same conditions. For these, we first uncover the neuroscientific perspective of LDM-based methods that is top-down creation based on pre-trained knowledge from massive images but lack of detail-driven bottom-up perception resulting in unfaithful details. We propose NeuralDiffuser which introduces primary visual feature guidance to provide detail cues in the form of gradients, extending the bottom-up process for LDM-based methods to achieve faithful semantics and details. We also developed a novel guidance strategy to ensure the consistency of repeated reconstructions rather than a
    
[^5]: DEFormer: 用于低光图像和暗视觉的DCT驱动增强Transformer

    DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision. (arXiv:2309.06941v1 [cs.CV])

    [http://arxiv.org/abs/2309.06941](http://arxiv.org/abs/2309.06941)

    该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。

    

    低光图像增强的目标是恢复图像的颜色和细节，对于自动驾驶中的高级视觉任务非常重要。然而，仅依靠RGB领域很难恢复暗区域的丢失细节。本文将频率作为网络的新线索，并提出了一种新颖的DCT驱动增强Transformer（DEFormer）。首先，我们提出了一个可学习的频率分支（LFB）用于频率增强，包括DCT处理和基于曲率的频率增强（CFE）。CFE计算每个通道的曲率以表示不同频率带的细节丰富度，然后我们将频率特征划分为更丰富纹理的频率带。此外，我们提出了一个交叉域融合（CDF）来减少RGB领域和频率领域之间的差异。我们还将DEFormer作为暗部检测的预处理，DEFormer有效提高了性能。

    The goal of low-light image enhancement is to restore the color and details of the image and is of great significance for high-level visual tasks in autonomous driving. However, it is difficult to restore the lost details in the dark area by relying only on the RGB domain. In this paper we introduce frequency as a new clue into the network and propose a novel DCT-driven enhancement transformer (DEFormer). First, we propose a learnable frequency branch (LFB) for frequency enhancement contains DCT processing and curvature-based frequency enhancement (CFE). CFE calculates the curvature of each channel to represent the detail richness of different frequency bands, then we divides the frequency features, which focuses on frequency bands with richer textures. In addition, we propose a cross domain fusion (CDF) for reducing the differences between the RGB domain and the frequency domain. We also adopt DEFormer as a preprocessing in dark detection, DEFormer effectively improves the performance
    
[^6]: 通过从心脏MRI中的知识转移解锁心电图的诊断潜力

    Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI. (arXiv:2308.05764v1 [eess.SP])

    [http://arxiv.org/abs/2308.05764](http://arxiv.org/abs/2308.05764)

    该论文提出了一种通过从心脏MRI中的知识转移解锁心电图的诊断潜力的方法。通过将CMR图像中的领域特定信息转移到ECG嵌入中，该方法实现了仅根据ECG数据进行全面的心脏筛查，并能预测心血管疾病的个体风险和确定心脏表型。

    

    心电图 (ECG) 是一种广泛可用的诊断工具，可以快速和经济高效地评估心血管健康状况。然而，在心血管疾病的诊断中，通常更喜欢使用昂贵的心脏磁共振 (CMR) 成像进行更详细的检查。虽然 CMR 成像可以提供详细的心脏解剖可视化，但由于长时间扫描和高昂的费用，它并不广泛可用。为了解决这个问题，我们提出了一种第一种自监督对比方法，将CMR图像中的领域特定信息转移到ECG嵌入中。我们的方法将多模态对比学习与屏蔽数据建模相结合，实现了仅根据ECG数据进行全面的心脏筛查。在使用来自40044名UK Biobank受试者的数据进行的广泛实验证明了我们方法的实用性和可推广性。我们预测了各种心血管疾病的个体风险，并仅根据ECG数据确定了不同的心脏表型。

    The electrocardiogram (ECG) is a widely available diagnostic tool that allows for a cost-effective and fast assessment of the cardiovascular health. However, more detailed examination with expensive cardiac magnetic resonance (CMR) imaging is often preferred for the diagnosis of cardiovascular diseases. While providing detailed visualization of the cardiac anatomy, CMR imaging is not widely available due to long scan times and high costs. To address this issue, we propose the first self-supervised contrastive approach that transfers domain-specific information from CMR images to ECG embeddings. Our approach combines multimodal contrastive learning with masked data modeling to enable holistic cardiac screening solely from ECG data. In extensive experiments using data from 40,044 UK Biobank subjects, we demonstrate the utility and generalizability of our method. We predict the subject-specific risk of various cardiovascular diseases and determine distinct cardiac phenotypes solely from E
    

