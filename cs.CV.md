# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data](https://arxiv.org/abs/2403.13106) | 该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。 |
| [^2] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^3] | [Effective Data Augmentation With Diffusion Models.](http://arxiv.org/abs/2302.07944) | 本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换方法，用于解决数据增强的多样性不足问题，并能够泛化到新视觉概念，从而提高了少样本图像分类和图像识别的性能。 |

# 详细

[^1]: 认识你的非线性：Shapley互动揭示数据的潜在结构

    Knowing Your Nonlinearities: Shapley Interactions Reveal the Underlying Structure of Data

    [https://arxiv.org/abs/2403.13106](https://arxiv.org/abs/2403.13106)

    该论文使用Shapley Taylor互动指数（STII）分析了底层数据结构对各种模态、任务和架构中模型表征的影响，发现了语言模型和语音模型中的新颖现象，并展示了特征交互如何直观反映对象边界。

    

    测量非线性特征交互是理解许多模型中复杂归因模式的一种已建立的方法。本文使用Shapley Taylor互动指数（STII）来分析底层数据结构对多种模态、任务和架构中模型表征的影响。在考虑掩码和自回归语言模型（MLMs和ALMs）中的语言结构时，我们发现STII在惯用表达中增加，MLMs随句法距离扩展STII，更多地依赖语法在其非线性结构中相比ALMs。我们的语音模型研究反映了口腔张开程度决定音素根据上下文变化的数量的原则。最后，我们研究图像分类器并说明特征交互直观反映对象边界。我们广泛的结果展示了跨学科工作和领域之间的益处。

    arXiv:2403.13106v1 Announce Type: cross  Abstract: Measuring nonlinear feature interaction is an established approach to understanding complex patterns of attribution in many models. In this paper, we use Shapley Taylor interaction indices (STII) to analyze the impact of underlying data structure on model representations in a variety of modalities, tasks, and architectures. Considering linguistic structure in masked and auto-regressive language models (MLMs and ALMs), we find that STII increases within idiomatic expressions and that MLMs scale STII with syntactic distance, relying more on syntax in their nonlinear structure than ALMs do. Our speech model findings reflect the phonetic principal that the openness of the oral cavity determines how much a phoneme varies based on its context. Finally, we study image classifiers and illustrate that feature interactions intuitively reflect object boundaries. Our wide range of results illustrates the benefits of interdisciplinary work and doma
    
[^2]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^3]: 利用扩散模型进行有效的数据增强

    Effective Data Augmentation With Diffusion Models. (arXiv:2302.07944v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.07944](http://arxiv.org/abs/2302.07944)

    本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换方法，用于解决数据增强的多样性不足问题，并能够泛化到新视觉概念，从而提高了少样本图像分类和图像识别的性能。

    

    数据增强是深度学习中最常见的工具之一，支撑着最近包括分类、生成模型和表示学习在内的许多进展。然而，当前的增强方法在数据的关键语义轴上缺乏多样性，缺乏改变高级语义属性（如场景中的动物种类）以增强数据多样性的方法。本文提出了一种利用预训练文本至图像扩散模型参数化的图像到图像转换来解决数据增强多样性不足问题的方法。我们的方法利用现成的扩散模型编辑图像，改变它们的语义，能够泛化到仅用少量标记示例得到的新视觉概念。我们在少样本图像分类任务和真实世界的杂草识别任务中评估了我们的方法，并观察到......

    Data augmentation is one of the most prevalent tools in deep learning, underpinning many recent advances, including those from classification, generative models, and representation learning. The standard approach to data augmentation combines simple transformations like rotations and flips to generate new images from existing ones. However, these new images lack diversity along key semantic axes present in the data. Current augmentations cannot alter the high-level semantic attributes, such as animal species present in a scene, to enhance the diversity of data. We address the lack of diversity in data augmentation with image-to-image transformations parameterized by pre-trained text-to-image diffusion models. Our method edits images to change their semantics using an off-the-shelf diffusion model, and generalizes to novel visual concepts from a few labelled examples. We evaluate our approach on few-shot image classification tasks, and on a real-world weed recognition task, and observe 
    

