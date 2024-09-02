# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane](https://arxiv.org/abs/2403.16210) | Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。 |

# 详细

[^1]: Frankenstein: 在一个三面位平面中生成语义-组合式3D场景

    Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane

    [https://arxiv.org/abs/2403.16210](https://arxiv.org/abs/2403.16210)

    Frankenstein是一个框架，可以在单个通道中同时生成多个语义相关的3D形状，为生成房间内部和人类化身等场景提供了有希望的结果。

    

    我们提出了Frankenstein，这是一个基于扩散的框架，可以在单个通道中生成语义-组合式3D场景。与现有方法输出单个统一的3D形状不同，Frankenstein同时生成多个独立的形状，每个对应一个语义上有意义的部分。3D场景信息编码在一个三面位平面张量中，从中可以解码多个符号距离函数（SDF）场以表示组合形状。在训练期间，一个自编码器将三面位平面压缩到潜在空间，然后使用去噪扩散过程来逼近组合场景的分布。Frankenstein在生成房间内部和具有自动分离部分的人类化身方面表现出有希望的结果。生成的场景有助于许多下游应用，例如部分重贴图、房间或化身衣服的对象重新排列。

    arXiv:2403.16210v1 Announce Type: cross  Abstract: We present Frankenstein, a diffusion-based framework that can generate semantic-compositional 3D scenes in a single pass. Unlike existing methods that output a single, unified 3D shape, Frankenstein simultaneously generates multiple separated shapes, each corresponding to a semantically meaningful part. The 3D scene information is encoded in one single tri-plane tensor, from which multiple Singed Distance Function (SDF) fields can be decoded to represent the compositional shapes. During training, an auto-encoder compresses tri-planes into a latent space, and then the denoising diffusion process is employed to approximate the distribution of the compositional scenes. Frankenstein demonstrates promising results in generating room interiors as well as human avatars with automatically separated parts. The generated scenes facilitate many downstream applications, such as part-wise re-texturing, object rearrangement in the room or avatar clo
    

