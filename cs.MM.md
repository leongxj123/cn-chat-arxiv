# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High Perceptual Quality Wireless Image Delivery with Denoising Diffusion Models.](http://arxiv.org/abs/2309.15889) | 本论文研究了通过深度学习的联合源-信道编码和去噪扩散模型在噪声无线信道上进行图像传输的问题。通过利用范围-零空间分解和逐步优化零空间内容，实现了在失真和感知质量方面的显著改进。 |

# 详细

[^1]: 使用去噪扩散模型实现高感知质量的无线图像传输

    High Perceptual Quality Wireless Image Delivery with Denoising Diffusion Models. (arXiv:2309.15889v1 [eess.IV])

    [http://arxiv.org/abs/2309.15889](http://arxiv.org/abs/2309.15889)

    本论文研究了通过深度学习的联合源-信道编码和去噪扩散模型在噪声无线信道上进行图像传输的问题。通过利用范围-零空间分解和逐步优化零空间内容，实现了在失真和感知质量方面的显著改进。

    

    我们考虑通过基于深度学习的联合源-信道编码（DeepJSCC）以及接收端的去噪扩散概率模型（DDPM）在噪声无线信道上进行图像传输。我们特别关注在实际有限块长度的情况下的感知失真权衡问题，这种情况下，分离的源编码和信道编码可能会高度不理想。我们引入了一种利用目标图像的范围-零空间分解的新方案。我们在编码后传输图像的范围空间，并使用DDPM逐步优化其零空间内容。通过广泛的实验证明，与标准的DeepJSCC和最先进的生成式学习方法相比，我们在重构图像的失真和感知质量方面实现了显著改进。为了促进进一步的研究和可重现性，我们将公开分享我们的源代码。

    We consider the image transmission problem over a noisy wireless channel via deep learning-based joint source-channel coding (DeepJSCC) along with a denoising diffusion probabilistic model (DDPM) at the receiver. Specifically, we are interested in the perception-distortion trade-off in the practical finite block length regime, in which separate source and channel coding can be highly suboptimal. We introduce a novel scheme that utilizes the range-null space decomposition of the target image. We transmit the range-space of the image after encoding and employ DDPM to progressively refine its null space contents. Through extensive experiments, we demonstrate significant improvements in distortion and perceptual quality of reconstructed images compared to standard DeepJSCC and the state-of-the-art generative learning-based method. We will publicly share our source code to facilitate further research and reproducibility.
    

