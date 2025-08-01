# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GANs Settle Scores!.](http://arxiv.org/abs/2306.01654) | 这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。 |
| [^2] | [Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots.](http://arxiv.org/abs/2304.01430) | 该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。 |

# 详细

[^1]: GANs解决分数争议问题！

    GANs Settle Scores!. (arXiv:2306.01654v1 [cs.LG])

    [http://arxiv.org/abs/2306.01654](http://arxiv.org/abs/2306.01654)

    这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。

    

    生成对抗网络（GAN）由一个生成器和一个判别器组成，生成器被训练以学习期望数据的基础分布，而判别器则被训练以区分真实样本和生成器输出的样本。本文提出了一种统一的方法，通过变分方法来分析生成器优化。在f-散度最小化 GAN 中，我们表明最优生成器是通过将其输出分布的得分与数据分布的得分进行匹配得到的。在IPM GAN中，我们表明这个最优生成器匹配得分型函数，包括与所选IPM约束空间相关的核流场。此外，IPM-GAN优化可以看作是平滑分数匹配中的一种，其中数据和生成器分布的得分与在核函数上进行卷积处理。

    Generative adversarial networks (GANs) comprise a generator, trained to learn the underlying distribution of the desired data, and a discriminator, trained to distinguish real samples from those output by the generator. A majority of GAN literature focuses on understanding the optimality of the discriminator through integral probability metric (IPM) or divergence based analysis. In this paper, we propose a unified approach to analyzing the generator optimization through variational approach. In $f$-divergence-minimizing GANs, we show that the optimal generator is the one that matches the score of its output distribution with that of the data distribution, while in IPM GANs, we show that this optimal generator matches score-like functions, involving the flow-field of the kernel associated with a chosen IPM constraint space. Further, the IPM-GAN optimization can be seen as one of smoothed score-matching, where the scores of the data and the generator distributions are convolved with the 
    
[^2]: 分离的关注力：基于上下文分离槽的无监督多对象发现

    Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots. (arXiv:2304.01430v1 [cs.CV])

    [http://arxiv.org/abs/2304.01430](http://arxiv.org/abs/2304.01430)

    该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。

    

    我们提出了一种将视觉场分割为独立运动区域的方法，不需要任何基础真值或监督。它由基于槽关注的对抗条件编码器-解码器架构组成，修改为使用图像作为上下文来解码光流，而不是尝试重构图像本身。在结果的多模式表示中，一种模式（流）将馈送给编码器以产生单独的潜在代码（槽），而另一种模式（图像）将决定解码器从槽生成第一个模式（流）。由于惯常的自编码基于最小化重构误差，并不能防止整个流被编码到一个槽中，因此我们将损失修改为基于上下文信息分离的对抗性标准。

    We introduce a method to segment the visual field into independently moving regions, trained with no ground truth or supervision. It consists of an adversarial conditional encoder-decoder architecture based on Slot Attention, modified to use the image as context to decode optical flow without attempting to reconstruct the image itself. In the resulting multi-modal representation, one modality (flow) feeds the encoder to produce separate latent codes (slots), whereas the other modality (image) conditions the decoder to generate the first (flow) from the slots. This design frees the representation from having to encode complex nuisance variability in the image due to, for instance, illumination and reflectance properties of the scene. Since customary autoencoding based on minimizing the reconstruction error does not preclude the entire flow from being encoded into a single slot, we modify the loss to an adversarial criterion based on Contextual Information Separation. The resulting min-m
    

