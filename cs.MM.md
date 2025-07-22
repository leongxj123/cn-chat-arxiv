# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing.](http://arxiv.org/abs/2306.16894) | PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。 |

# 详细

[^1]: PFB-Diff: 渐进特征混合扩散用于文本驱动的图像编辑

    PFB-Diff: Progressive Feature Blending Diffusion for Text-driven Image Editing. (arXiv:2306.16894v1 [cs.CV])

    [http://arxiv.org/abs/2306.16894](http://arxiv.org/abs/2306.16894)

    PFB-Diff 是一个通过渐进特征混合的方法，用于文本驱动的图像编辑。该方法解决了扩散模型在像素级混合中产生的伪影问题，并通过多级特征混合和注意力屏蔽机制确保了编辑图像的语义连贯性和高质量。

    

    扩散模型展示了其合成多样性和高质量图像的卓越能力，引起了人们对将其应用于实际图像编辑的兴趣。然而，现有的基于扩散的局部图像编辑方法常常因为目标图像和扩散潜在变量的像素级混合而产生不期望的伪影，缺乏维持图像一致性所必需的语义。为了解决这些问题，我们提出了PFB-Diff，一种逐步特征混合的方法，用于基于扩散的图像编辑。与以往方法不同，PFB-Diff通过多级特征混合将文本引导生成的内容与目标图像无缝集成在一起。深层特征中编码的丰富语义和从高到低级别的渐进混合方案确保了编辑图像的语义连贯性和高质量。此外，我们在交叉注意力层中引入了一个注意力屏蔽机制，以限制特定词语对编辑图像的影响。

    Diffusion models have showcased their remarkable capability to synthesize diverse and high-quality images, sparking interest in their application for real image editing. However, existing diffusion-based approaches for local image editing often suffer from undesired artifacts due to the pixel-level blending of the noised target images and diffusion latent variables, which lack the necessary semantics for maintaining image consistency. To address these issues, we propose PFB-Diff, a Progressive Feature Blending method for Diffusion-based image editing. Unlike previous methods, PFB-Diff seamlessly integrates text-guided generated content into the target image through multi-level feature blending. The rich semantics encoded in deep features and the progressive blending scheme from high to low levels ensure semantic coherence and high quality in edited images. Additionally, we introduce an attention masking mechanism in the cross-attention layers to confine the impact of specific words to 
    

