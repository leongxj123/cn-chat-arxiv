# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Natural Counterfactuals With Necessary Backtracking](https://rss.arxiv.org/abs/2402.01607) | 本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。 |
| [^2] | [Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level](https://arxiv.org/abs/2403.04690) | 该研究提出了一种更快的邻域注意力机制，通过将注意力限制在最近的邻居之间来降低自注意力的计算复杂度，实现了显著的性能提升。 |
| [^3] | [NM-FlowGAN: Modeling sRGB Noise with a Hybrid Approach based on Normalizing Flows and Generative Adversarial Networks](https://arxiv.org/abs/2312.10112) | NM-FlowGAN是一种利用生成对抗网络和正规化流的混合方法，旨在更准确地建模sRGB噪声，弥补了单一生成模型固有特性所带来的性能限制。 |
| [^4] | [Invisible Image Watermarks Are Provably Removable Using Generative AI.](http://arxiv.org/abs/2306.01953) | 该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。 |
| [^5] | [Pre-trained transformer for adversarial purification.](http://arxiv.org/abs/2306.01762) | 本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。 |

# 详细

[^1]: 具有必要回溯的自然反事实

    Natural Counterfactuals With Necessary Backtracking

    [https://rss.arxiv.org/abs/2402.01607](https://rss.arxiv.org/abs/2402.01607)

    本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。

    

    反事实推理对于人类认知非常重要，尤其对于提供解释和做出决策至关重要。尽管Judea Pearl的研究方法在理论上很优雅，但其生成反事实情景往往需要过于脱离实际情景的干预，因此难以实施。为了解决这个问题，我们提出了一种自然反事实的框架和一种根据实际世界数据分布生成自然反事实的方法。我们的方法提供了对反事实推理的改进，允许对因果前置变量进行改变以最小化与实际情景的偏差。为了生成自然反事实，我们引入了一种创新的优化框架，通过自然性准则允许但控制回溯的范围。实证实验表明了我们方法的有效性。

    Counterfactual reasoning is pivotal in human cognition and especially important for providing explanations and making decisions. While Judea Pearl's influential approach is theoretically elegant, its generation of a counterfactual scenario often requires interventions that are too detached from the real scenarios to be feasible. In response, we propose a framework of natural counterfactuals and a method for generating counterfactuals that are natural with respect to the actual world's data distribution. Our methodology refines counterfactual reasoning, allowing changes in causally preceding variables to minimize deviations from realistic scenarios. To generate natural counterfactuals, we introduce an innovative optimization framework that permits but controls the extent of backtracking with a naturalness criterion. Empirical experiments indicate the effectiveness of our method.
    
[^2]: 更快的邻域注意力: 在线程块级别减少自注意力的O(n^2)成本

    Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level

    [https://arxiv.org/abs/2403.04690](https://arxiv.org/abs/2403.04690)

    该研究提出了一种更快的邻域注意力机制，通过将注意力限制在最近的邻居之间来降低自注意力的计算复杂度，实现了显著的性能提升。

    

    邻域注意力通过限制每个标记的注意力范围为其最近的邻居来降低自注意力的成本。该限制由窗口大小和扩张因子参数化，介于线性投影和自注意力之间绘制了可能的注意力模式谱。邻域注意力，以及更一般地滑动窗口注意力模式，在基础设施方面长期受到限制，特别是在更高秩的空间（2-D和3-D），促使开发定制内核的发展，这些内核在功能或性能方面受限，如果不是两者都有。在这项工作中，我们首先展示邻域注意力可以表示为批量化的GEMM问题，类似于标准注意力，并为1-D和2-D邻域注意力实现它。与现有的简单内核相比，这些内核平均提供了分别是1-D和2-D邻域注意力的全精度延迟改进分别为895%和272%。

    arXiv:2403.04690v1 Announce Type: cross  Abstract: Neighborhood attention reduces the cost of self attention by restricting each token's attention span to its nearest neighbors. This restriction, parameterized by a window size and dilation factor, draws a spectrum of possible attention patterns between linear projection and self attention. Neighborhood attention, and more generally sliding window attention patterns, have long been bounded by infrastructure, particularly in higher-rank spaces (2-D and 3-D), calling for the development of custom kernels, which have been limited in either functionality, or performance, if not both. In this work, we first show that neighborhood attention can be represented as a batched GEMM problem, similar to standard attention, and implement it for 1-D and 2-D neighborhood attention. These kernels on average provide 895% and 272% improvement in full precision latency compared to existing naive kernels for 1-D and 2-D neighborhood attention respectively. 
    
[^3]: NM-FlowGAN: 基于正规化流和生成对抗网络的混合方法对sRGB噪声进行建模

    NM-FlowGAN: Modeling sRGB Noise with a Hybrid Approach based on Normalizing Flows and Generative Adversarial Networks

    [https://arxiv.org/abs/2312.10112](https://arxiv.org/abs/2312.10112)

    NM-FlowGAN是一种利用生成对抗网络和正规化流的混合方法，旨在更准确地建模sRGB噪声，弥补了单一生成模型固有特性所带来的性能限制。

    

    建模和合成真实的sRGB噪声对于各种低级别视觉任务至关重要，例如构建用于训练图像去噪系统的数据集。真实sRGB噪声的分布极为复杂，并受多种因素影响，使得其准确建模极具挑战性。因此，最近的研究提出了采用数据驱动生成模型，如生成对抗网络（GAN）和正规化流的方法。这些研究相比传统的噪声建模方法实现了对sRGB噪声的更准确建模。然而，由于每种生成模型的固有特性，存在性能限制。为了解决这个问题，我们提出了NM-FlowGAN，这是一种利用GAN和正规化流的优势的混合方法。我们同时采用基于正规化流的像素级噪声建模网络，以及基于GAN的空间相关性建模网络。

    arXiv:2312.10112v2 Announce Type: replace-cross  Abstract: Modeling and synthesizing real sRGB noise is crucial for various low-level vision tasks, such as building datasets for training image denoising systems. The distribution of real sRGB noise is highly complex and affected by a multitude of factors, making its accurate modeling extremely challenging. Therefore, recent studies have proposed methods that employ data-driven generative models, such as generative adversarial networks (GAN) and Normalizing Flows. These studies achieve more accurate modeling of sRGB noise compared to traditional noise modeling methods. However, there are performance limitations due to the inherent characteristics of each generative model. To address this issue, we propose NM-FlowGAN, a hybrid approach that exploits the strengths of both GAN and Normalizing Flows. We simultaneously employ a pixel-wise noise modeling network based on Normalizing Flows, and spatial correlation modeling networks based on GAN
    
[^4]: 通过生成式AI，证明了隐形图像水印是可清除的

    Invisible Image Watermarks Are Provably Removable Using Generative AI. (arXiv:2306.01953v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.01953](http://arxiv.org/abs/2306.01953)

    该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。

    

    隐形水印通过嵌入只有权利拥有者可以检测到的隐藏信息来保护图像的版权。它们还防止人们滥用由AI模型生成的图像。我们提出了一种家族化再生攻击来清除这些隐形水印。所提出的攻击方法首先向图像添加随机噪声来破坏水印，然后重建图像。这种方法灵活，可以与许多现有的图像降噪算法和预训练的生成模型（如扩散模型）实例化。通过形式化证明和实证结果，我们证明了所有隐形水印都容易受到所提出的攻击。对于一个特别有弹性的水印RivaGAN，再生攻击可以去除93-99%的隐形水印，而基线攻击只能去除不超过3%。然而，如果我们不要求带水印的图像与原始图像相同，保持图像语义相似的水印可能是一种替代方案。

    Invisible watermarks safeguard images' copyright by embedding hidden messages only detectable by owners. They also prevent people from misusing images, especially those generated by AI models. We propose a family of regeneration attacks to remove these invisible watermarks. The proposed attack method first adds random noise to an image to destroy the watermark and then reconstructs the image. This approach is flexible and can be instantiated with many existing image-denoising algorithms and pre-trained generative models such as diffusion models. Through formal proofs and empirical results, we show that all invisible watermarks are vulnerable to the proposed attack. For a particularly resilient watermark, RivaGAN, regeneration attacks remove 93-99% of the invisible watermarks while the baseline attacks remove no more than 3%. However, if we do not require the watermarked image to look the same as the original one, watermarks that keep the image semantically similar can be an alternative
    
[^5]: 预训练Transformer用于对抗性样本提纯

    Pre-trained transformer for adversarial purification. (arXiv:2306.01762v1 [cs.CR])

    [http://arxiv.org/abs/2306.01762](http://arxiv.org/abs/2306.01762)

    本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。

    

    随着越来越多的深度神经网络被部署为各种日常服务，它们的可靠性至关重要。深度神经网络容易受到对抗性攻击的影响，其中逃避攻击是最普遍的一种。最近的研究通常通过对抗训练或利用大量清洁数据的知识来增强其健壮性。然而，在实际应用中，重新训练和部署模型需要大量的计算资源，对在线服务造成重大损失。此外，当检测到某种攻击的对抗性例子时，服务提供者只能获得有限的对抗性样本，而大量的清洁数据可能无法获取。针对这些问题，我们提出了一种新的方案，名为RaPiD（Rapid Plug-in Defender），旨在快速防御具有少量干净和对抗性示例限制的原始服务模型的某种攻击。受到预训练模型提供转移学习良好初始化的通用趋势的启发，我们建议通过微调预先训练的Transformer来提纯对抗性样本。预训练的Transformer作为正则化器，鼓励提纯后的对抗性样本接近清晰数据的分布。实验结果表明，RaPiD在防御各种具有限数据的攻击方面优于最先进的方法。

    With more and more deep neural networks being deployed as various daily services, their reliability is essential. It's frightening that deep neural networks are vulnerable and sensitive to adversarial attacks, the most common one of which for the services is evasion-based. Recent works usually strengthen the robustness by adversarial training or leveraging the knowledge of an amount of clean data. However, in practical terms, retraining and redeploying the model need a large computational budget, leading to heavy losses to the online service. In addition, when adversarial examples of a certain attack are detected, only limited adversarial examples are available for the service provider, while much clean data may not be accessible. Given the mentioned problems, we propose a new scenario, RaPiD (Rapid Plug-in Defender), which is to rapidly defend against a certain attack for the frozen original service model with limitations of few clean and adversarial examples. Motivated by the general
    

