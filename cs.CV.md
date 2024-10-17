# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers](https://arxiv.org/abs/2402.02263) | MixedNUTS是一种无需训练的方法，通过非线性混合分类器的转换和概率混合来实现准确性和鲁棒性的平衡。 |
| [^2] | [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769) | AnimateLCM提出了一种分离的一致性学习策略，通过将图像生成优先级和动作生成优先级的蒸馏分离开来，提高了训练效率并增强了生成的视觉质量。 |
| [^3] | [Reverse Stable Diffusion: What prompt was used to generate this image?.](http://arxiv.org/abs/2308.01472) | 本论文介绍了一种新的任务，即在给定由生成扩散模型生成的图像的情况下预测文本提示。为了解决这个问题，作者结合了多种白盒和黑盒模型，提出了一个新颖的学习框架，该框架能够生成改进的提示，并采用课程学习和无监督领域自适应核学习方法来进一步提高方法的性能。 |
| [^4] | [DeepMSS: Deep Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images.](http://arxiv.org/abs/2305.09946) | 提出了一种DeepMSS模型，采用新颖的Segmentated-to-Survival（STS）框架，使用多模态渐进聚合网络（MMPAN）来探索肿瘤内外的预后信息，并通过自我注意力机制增强的深度生存模型进行生存预测，取得了在两个公共PET/CT图像数据集上优于几种最先进的方法的结果。 |

# 详细

[^1]: MixedNUTS: 通过非线性混合分类器实现无需训练的准确性和鲁棒性平衡

    MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers

    [https://arxiv.org/abs/2402.02263](https://arxiv.org/abs/2402.02263)

    MixedNUTS是一种无需训练的方法，通过非线性混合分类器的转换和概率混合来实现准确性和鲁棒性的平衡。

    

    鲁棒性往往牺牲了准确性，阻碍了鲁棒分类模型在实际应用中的使用。基于训练的解决方案在与已训练的大型高性能模型兼容性方面存在限制，因此需要探索无需训练的集成方法。我们观察到鲁棒模型在干净数据和对抗数据上的正确预测比错误预测更自信，我们推测通过增强这种“良性置信度特性”可以在集成环境中实现准确性和鲁棒性的平衡。为了实现这一点，我们提出了“MixedNUTS”，一种无需训练的方法，利用仅有三个参数的非线性转换来处理鲁棒分类器和标准非鲁棒分类器的输出Logits，并通过高效算法进行优化。然后，MixedNUTS将转换后的Logits转换为概率，并将它们混合作为最终的输出。在CIFAR-10、CIFAR-100和ImageNet数据集上进行了实验。

    Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet da
    
[^2]: AnimateLCM: 使用分离的一致性学习加速个性化的扩散模型和适配器的动画生成

    AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning

    [https://arxiv.org/abs/2402.00769](https://arxiv.org/abs/2402.00769)

    AnimateLCM提出了一种分离的一致性学习策略，通过将图像生成优先级和动作生成优先级的蒸馏分离开来，提高了训练效率并增强了生成的视觉质量。

    

    视频扩散模型因其能够产生连贯且高保真度的视频而受到越来越多的关注。然而，迭代的去噪过程使其计算密集且耗时，从而限制了其应用。受一致性模型（CM）的启发，该模型通过最小的步骤蒸馏预训练的图像扩散模型以加速采样，以及其在条件图像生成上的成功扩展——潜在一致性模型（LCM），我们提出了AnimateLCM，允许在最小的步骤内生成高保真度的视频。我们提出了一种分离的一致性学习策略，将图像生成优先级和动作生成优先级的蒸馏分离开来，这提高了训练效率并增强了生成的视觉质量。此外，为了实现稳定的扩散社区中的即插即用适配器的组合以实现各种修改，我们还引入了适配器的概念。

    Video diffusion models has been gaining increasing attention for its ability to produce videos that are both coherent and of high fidelity. However, the iterative denoising process makes it computationally intensive and time-consuming, thus limiting its applications. Inspired by the Consistency Model (CM) that distills pretrained image diffusion models to accelerate the sampling with minimal steps and its successful extension Latent Consistency Model (LCM) on conditional image generation, we propose AnimateLCM, allowing for high-fidelity video generation within minimal steps. Instead of directly conducting consistency learning on the raw video dataset, we propose a decoupled consistency learning strategy that decouples the distillation of image generation priors and motion generation priors, which improves the training efficiency and enhance the generation visual quality. Additionally, to enable the combination of plug-and-play adapters in stable diffusion community to achieve various 
    
[^3]: 反向稳定扩散：生成该图像所使用的提示是什么？

    Reverse Stable Diffusion: What prompt was used to generate this image?. (arXiv:2308.01472v1 [cs.CV])

    [http://arxiv.org/abs/2308.01472](http://arxiv.org/abs/2308.01472)

    本论文介绍了一种新的任务，即在给定由生成扩散模型生成的图像的情况下预测文本提示。为了解决这个问题，作者结合了多种白盒和黑盒模型，提出了一个新颖的学习框架，该框架能够生成改进的提示，并采用课程学习和无监督领域自适应核学习方法来进一步提高方法的性能。

    

    文本到图像扩散模型，如稳定扩散，最近吸引了许多研究人员的兴趣，反向扩散过程在更好地理解生成过程和如何设计提示以获得所需图像方面起着重要作用。为此，我们引入了一种新的任务，即在给定由生成扩散模型生成的图像的情况下预测文本提示。我们结合了一系列白盒和黑盒模型（有和无对扩散网络权重进行访问）来处理所提出的任务。我们提出了一个新颖的学习框架，包括联合提示回归和多标签词汇分类目标，生成改进的提示。为了进一步改进我们的方法，我们采用了一个课程学习过程，促进了具有更低标注噪声（即更好对齐）的图像提示对的学习，并且使用相似性进行无监督领域自适应核学习方法。

    Text-to-image diffusion models such as Stable Diffusion have recently attracted the interest of many researchers, and inverting the diffusion process can play an important role in better understanding the generative process and how to engineer prompts in order to obtain the desired images. To this end, we introduce the new task of predicting the text prompt given an image generated by a generative diffusion model. We combine a series of white-box and black-box models (with and without access to the weights of the diffusion network) to deal with the proposed task. We propose a novel learning framework comprising of a joint prompt regression and multi-label vocabulary classification objective that generates improved prompts. To further improve our method, we employ a curriculum learning procedure that promotes the learning of image-prompt pairs with lower labeling noise (i.e. that are better aligned), and an unsupervised domain-adaptive kernel learning method that uses the similarities b
    
[^4]: DeepMSS：基于PET/CT图像的深度多模态切片到生存预测学习

    DeepMSS: Deep Multi-Modality Segmentation-to-Survival Learning for Survival Outcome Prediction from PET/CT Images. (arXiv:2305.09946v1 [eess.IV])

    [http://arxiv.org/abs/2305.09946](http://arxiv.org/abs/2305.09946)

    提出了一种DeepMSS模型，采用新颖的Segmentated-to-Survival（STS）框架，使用多模态渐进聚合网络（MMPAN）来探索肿瘤内外的预后信息，并通过自我注意力机制增强的深度生存模型进行生存预测，取得了在两个公共PET/CT图像数据集上优于几种最先进的方法的结果。

    

    生存预测是癌症管理的主要关注点。基于深度学习的深度生存模型已被广泛采用，用于在医学图像上执行端到端的生存预测。最近的深度生存模型通过联合执行肿瘤分割和生存预测，采用多任务学习指导模型提取与肿瘤相关的信息，取得了有希望的性能。然而，现有的深度生存模型在探索肿瘤外预后信息（例如，局部淋巴结转移和邻近组织侵袭）方面存在困难。此外，现有的深度生存模型在利用多模态图像方面欠发展。为了解决这些问题，我们提出了一种名为DeepMSS的深度多模态切片到生存模型。该模型采用一种新颖的Segmentated-to-Survival（STS）框架，通过分离分割和生存预测任务来进行。对于分割，我们使用一种新颖的多模态渐进聚合网络（MMPAN）来探索肿瘤内外的预后信息。对于生存预测，我们提出了一种自我注意力机制增强的深度生存模型，该模型学习MMPAN的特征表示并执行生存预测。在两个公共PET/CT图像数据集上的实验结果表明，我们提出的DeepMSS模型在生存预测方面优于几种最先进的方法。

    Survival prediction is a major concern for cancer management. Deep survival models based on deep learning have been widely adopted to perform end-to-end survival prediction from medical images. Recent deep survival models achieved promising performance by jointly performing tumor segmentation with survival prediction, where the models were guided to extract tumor-related information through Multi-Task Learning (MTL). However, existing deep survival models have difficulties in exploring out-of-tumor prognostic information (e.g., local lymph node metastasis and adjacent tissue invasions). In addition, existing deep survival models are underdeveloped in utilizing multi-modality images. Empirically-designed strategies were commonly adopted to fuse multi-modality information via fixed pre-designed networks. In this study, we propose a Deep Multi-modality Segmentation-to-Survival model (DeepMSS) for survival prediction from PET/CT images. Instead of adopting MTL, we propose a novel Segmentat
    

