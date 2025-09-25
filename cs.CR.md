# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks](https://arxiv.org/abs/2402.06357) | 本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。 |
| [^2] | [Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies.](http://arxiv.org/abs/2210.06140) | 本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。 |

# 详细

[^1]: SpongeNet 攻击：深度神经网络的海绵权重中毒

    The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks

    [https://arxiv.org/abs/2402.06357](https://arxiv.org/abs/2402.06357)

    本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。

    

    海绵攻击旨在增加在硬件加速器上部署的神经网络的能耗和计算时间。现有的海绵攻击可以通过海绵示例进行推理，也可以通过海绵中毒在训练过程中进行。海绵示例利用添加到模型输入的扰动来增加能量和延迟，而海绵中毒则改变模型的目标函数来引发推理时的能量/延迟效应。在这项工作中，我们提出了一种新颖的海绵攻击，称为 SpongeNet。SpongeNet 是第一个直接作用于预训练模型参数的海绵攻击。我们的实验表明，相比于海绵中毒，SpongeNet 可以成功增加视觉模型的能耗，并且所需的样本数量更少。我们的实验结果表明，如果不专门针对海绵中毒进行调整（即减小批归一化偏差值），则毒害防御会失效。我们的工作显示出海绵攻击的影响。

    Sponge attacks aim to increase the energy consumption and computation time of neural networks deployed on hardware accelerators. Existing sponge attacks can be performed during inference via sponge examples or during training via Sponge Poisoning. Sponge examples leverage perturbations added to the model's input to increase energy and latency, while Sponge Poisoning alters the objective function of a model to induce inference-time energy/latency effects.   In this work, we propose a novel sponge attack called SpongeNet. SpongeNet is the first sponge attack that is performed directly on the parameters of a pre-trained model. Our experiments show that SpongeNet can successfully increase the energy consumption of vision models with fewer samples required than Sponge Poisoning. Our experiments indicate that poisoning defenses are ineffective if not adjusted specifically for the defense against Sponge Poisoning (i.e., they decrease batch normalization bias values). Our work shows that Spong
    
[^2]: 差分隐私引导采样：新的隐私分析与推断策略

    Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies. (arXiv:2210.06140v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06140](http://arxiv.org/abs/2210.06140)

    本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。

    

    差分隐私机制通过引入随机性来保护个人信息，但在应用中，统计推断仍然缺乏通用技术。本文研究了一个差分隐私引导采样方法，通过发布多个私有引导采样估计来推断样本分布并构建置信区间。我们的隐私分析提供了单个差分隐私引导采样估计的隐私成本新结果，适用于任何差分隐私机制，并指出了现有文献中引导采样的一些误用。使用Gaussian-DP（GDP）框架，我们证明从满足 $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP 的机制中释放 $B$ 个差分隐私引导采样估计，在 $B$ 趋近无限大时渐近地满足 $\mu$-GDP。此外，我们使用差分隐私引导采样估计的反卷积对样本分布进行准确推断。

    Differentially private (DP) mechanisms protect individual-level information by introducing randomness into the statistical analysis procedure. Despite the availability of numerous DP tools, there remains a lack of general techniques for conducting statistical inference under DP. We examine a DP bootstrap procedure that releases multiple private bootstrap estimates to infer the sampling distribution and construct confidence intervals (CIs). Our privacy analysis presents new results on the privacy cost of a single DP bootstrap estimate, applicable to any DP mechanisms, and identifies some misapplications of the bootstrap in the existing literature. Using the Gaussian-DP (GDP) framework (Dong et al.,2022), we show that the release of $B$ DP bootstrap estimates from mechanisms satisfying $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP asymptotically satisfies $\mu$-GDP as $B$ goes to infinity. Moreover, we use deconvolution with the DP bootstrap estimates to accurately infer the sampling distribution
    

