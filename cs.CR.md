# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fingerprinting Generative Adversarial Networks.](http://arxiv.org/abs/2106.11760) | 本文提出了一种保护GAN知识产权的指纹识别方案，通过生成指纹样本并嵌入到分类器中进行版权验证，解决了前一种对分类模型的指纹识别方法在简单转移至GAN时遇到的隐蔽性和鲁棒性瓶颈，具有实际保护现代GAN模型的可行性。 |

# 详细

[^1]: 生成对抗网络的指纹识别技术

    Fingerprinting Generative Adversarial Networks. (arXiv:2106.11760v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2106.11760](http://arxiv.org/abs/2106.11760)

    本文提出了一种保护GAN知识产权的指纹识别方案，通过生成指纹样本并嵌入到分类器中进行版权验证，解决了前一种对分类模型的指纹识别方法在简单转移至GAN时遇到的隐蔽性和鲁棒性瓶颈，具有实际保护现代GAN模型的可行性。

    

    生成对抗网络（GANs）已经广泛应用于各种应用场景。由于商业GAN的生产需要大量的计算和人力资源，因此迫切需要版权保护。本文提出了一种用于保护GAN知识产权的指纹识别方案。我们突破了前一种对分类模型的指纹识别方法在简单转移至GAN时所遇到的隐蔽性和鲁棒性瓶颈。具体来说，我们创造性地从目标GAN和分类器构建一个复合深度学习模型。然后，我们从这个复合模型中产生指纹样本，并将其嵌入到分类器中，以进行有效的版权验证。这种方案启发了一些具体的方法，以实际保护现代GAN模型。理论分析证明了这些方法可以满足知识产权保护所需要的不同安全要求。我们还进行了实验来证明该方案的功效。

    Generative Adversarial Networks (GANs) have been widely used in various application scenarios. Since the production of a commercial GAN requires substantial computational and human resources, the copyright protection of GANs is urgently needed. In this paper, we present the first fingerprinting scheme for the Intellectual Property (IP) protection of GANs. We break through the stealthiness and robustness bottlenecks suffered by previous fingerprinting methods for classification models being naively transferred to GANs. Specifically, we innovatively construct a composite deep learning model from the target GAN and a classifier. Then we generate fingerprint samples from this composite model, and embed them in the classifier for effective ownership verification. This scheme inspires some concrete methodologies to practically protect the modern GAN models. Theoretical analysis proves that these methods can satisfy different security requirements necessary for IP protection. We also conduct 
    

