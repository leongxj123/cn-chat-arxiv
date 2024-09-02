# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane](https://arxiv.org/abs/2403.16210) | Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。 |
| [^2] | [Object-Centric Diffusion for Efficient Video Editing.](http://arxiv.org/abs/2401.05735) | 本论文提出了一种面向对象的扩散技术，通过分配更多的计算资源给前景编辑区域来实现视频编辑的高效率，从而大大提高了速度，同时保持了质量。 |
| [^3] | [Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm.](http://arxiv.org/abs/2310.13019) | 本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。 |
| [^4] | [Does CLIP Bind Concepts? Probing Compositionality in Large Image Models.](http://arxiv.org/abs/2212.10537) | 本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。 |

# 详细

[^1]: Frankenstein: 在一个三面位平面中生成语义-组合式3D场景

    Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane

    [https://arxiv.org/abs/2403.16210](https://arxiv.org/abs/2403.16210)

    Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。

    

    我们提出了Frankenstein，这是一个基于扩散的框架，可以在单个通道中生成语义-组合式3D场景。与现有方法输出单个统一的3D形状不同，Frankenstein同时生成多个独立的形状，每个对应一个语义上有意义的部分。3D场景信息编码在一个三面位平面张量中，从中可以解码多个符号距离函数（SDF）场以表示组合形状。在训练期间，一个自编码器将三面位平面压缩到潜在空间，然后使用去噪扩散过程来逼近组合场景的分布。Frankenstein在生成房间内部和具有自动分离部分的人类化身方面表现出有希望的结果。生成的场景有助于许多下游应用，例如部分重贴图、房间或化身衣服的对象重新排列。

    arXiv:2403.16210v1 Announce Type: cross  Abstract: We present Frankenstein, a diffusion-based framework that can generate semantic-compositional 3D scenes in a single pass. Unlike existing methods that output a single, unified 3D shape, Frankenstein simultaneously generates multiple separated shapes, each corresponding to a semantically meaningful part. The 3D scene information is encoded in one single tri-plane tensor, from which multiple Singed Distance Function (SDF) fields can be decoded to represent the compositional shapes. During training, an auto-encoder compresses tri-planes into a latent space, and then the denoising diffusion process is employed to approximate the distribution of the compositional scenes. Frankenstein demonstrates promising results in generating room interiors as well as human avatars with automatically separated parts. The generated scenes facilitate many downstream applications, such as part-wise re-texturing, object rearrangement in the room or avatar clo
    
[^2]: 面向对象的扩散技术实现高效视频编辑

    Object-Centric Diffusion for Efficient Video Editing. (arXiv:2401.05735v1 [cs.CV])

    [http://arxiv.org/abs/2401.05735](http://arxiv.org/abs/2401.05735)

    本论文提出了一种面向对象的扩散技术，通过分配更多的计算资源给前景编辑区域来实现视频编辑的高效率，从而大大提高了速度，同时保持了质量。

    

    基于扩散的视频编辑已经达到了令人印象深刻的质量，并且可以根据编辑提示来转换视频的全局风格、局部结构和属性。然而，这些解决方案通常需要使用大量的内存和计算资源来生成具有时序一致性的帧，可能涉及扩散反演和/或跨帧注意力。在本文中，我们对这种低效性进行了分析，并提出了简单而有效的修改，可以显著提高速度同时保持质量。此外，我们引入了面向对象的扩散技术（OCD），通过将计算资源更多地分配给对感知质量更重要的前景编辑区域，进一步降低延迟。我们通过两个新的提案来实现这一点：i）面向对象的采样，将用于显著区域或背景的扩散步骤与用于前景的扩散步骤分离开来，将大部分模型容量分配给前者；ii）面向对象的3D令牌合并，用于改善前景和背景之间的混合。

    Diffusion-based video editing have reached impressive quality and can transform either the global style, local structure, and attributes of given video inputs, following textual edit prompts. However, such solutions typically incur heavy memory and computational costs to generate temporally-coherent frames, either in the form of diffusion inversion and/or cross-frame attention. In this paper, we conduct an analysis of such inefficiencies, and suggest simple yet effective modifications that allow significant speed-ups whilst maintaining quality. Moreover, we introduce Object-Centric Diffusion, coined as OCD, to further reduce latency by allocating computations more towards foreground edited regions that are arguably more important for perceptual quality. We achieve this by two novel proposals: i) Object-Centric Sampling, decoupling the diffusion steps spent on salient regions or background, allocating most of the model capacity to the former, and ii) Object-Centric 3D Token Merging, whi
    
[^3]: 通过DeepFool算法对深度神经网络进行有针对性的类别操纵的对抗攻击定制

    Tailoring Adversarial Attacks on Deep Neural Networks for Targeted Class Manipulation Using DeepFool Algorithm. (arXiv:2310.13019v1 [cs.CV])

    [http://arxiv.org/abs/2310.13019](http://arxiv.org/abs/2310.13019)

    本文提出了一种增强版DeepFool算法，名为Targeted DeepFool，可以针对特定类别进行错误分类，并引入了最小置信度分数要求超参数来提高灵活性。

    

    深度神经网络（DNNs）在各个领域都取得了显著的进展，但对抗攻击的易受攻击性引起了严重关注。了解这些易受攻击性并开发有效的防御机制至关重要。DeepFool是Moosavi-Dezfooli等人（2016年）提出的一种算法，用于找到将输入图像错误分类的最小扰动。然而，DeepFool缺乏有针对性的方法，使其在特定攻击场景中的有效性较低。此外，在先前的相关工作中，研究人员主要关注的是成功率，而没有考虑图像被扭曲的程度、图像质量的完整性以及错误分类的置信度水平。因此，在本文中，我们提出了Targeted DeepFool，这是DeepFool的增强版，可以针对特定类别进行错误分类。我们还引入了一个最小置信度分数要求超参数来增强灵活性。我们的实验证明了所提方法在不同情况下的有效性和效率。

    Deep neural networks (DNNs) have significantly advanced various domains, but their vulnerability to adversarial attacks poses serious concerns. Understanding these vulnerabilities and developing effective defense mechanisms is crucial. DeepFool, an algorithm proposed by Moosavi-Dezfooli et al. (2016), finds minimal perturbations to misclassify input images. However, DeepFool lacks a targeted approach, making it less effective in specific attack scenarios. Also, in previous related works, researchers primarily focus on success, not considering how much an image is getting distorted; the integrity of the image quality, and the confidence level to misclassifying. So, in this paper, we propose Targeted DeepFool, an augmented version of DeepFool that allows targeting specific classes for misclassification. We also introduce a minimum confidence score requirement hyperparameter to enhance flexibility. Our experiments demonstrate the effectiveness and efficiency of the proposed method across 
    
[^4]: CLIP是否捆绑概念？探索大型图像模型的组合性。

    Does CLIP Bind Concepts? Probing Compositionality in Large Image Models. (arXiv:2212.10537v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.10537](http://arxiv.org/abs/2212.10537)

    本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。

    

    近年来，结合文本和图像的大型神经网络模型取得了令人瞩目的进展。然而，这些模型在多大程度上编码了它们操作的概念的组成性表示，如通过对“红色立方体”进行推理以正确识别“红色”和“立方体”这些成分，这仍然是一个开放性问题。本文关注一个大型预训练的视觉和语言模型（CLIP）编码组合概念的能力以及以结构敏感的方式捆绑变量的能力（例如区分“立方体在球体后面”和“球体在立方体后面”）。为了检查CLIP的性能，我们比较了许多来自组合分布语义模型（CDSMs）的架构，这是一种试图在嵌入空间中实现传统组合语言结构的研究方向。我们发现CLIP能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。我们的分析凸显了评估大型模型组合性的重要性，并为未来研究提出了方向。

    Large-scale neural network models combining text and images have made incredible progress in recent years. However, it remains an open question to what extent such models encode compositional representations of the concepts over which they operate, such as correctly identifying ''red cube'' by reasoning over the constituents ''red'' and ''cube''. In this work, we focus on the ability of a large pretrained vision and language model (CLIP) to encode compositional concepts and to bind variables in a structure-sensitive way (e.g., differentiating ''cube behind sphere'' from ''sphere behind cube''). In order to inspect the performance of CLIP, we compare several architectures from research on compositional distributional semantics models (CDSMs), a line of research that attempts to implement traditional compositional linguistic structures within embedding spaces. We find that CLIP can compose concepts in a single-object setting, but in situations where concept binding is needed, performance
    

