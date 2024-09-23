# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HeadEvolver: Text to Head Avatars via Locally Learnable Mesh Deformation](https://arxiv.org/abs/2403.09326) | 通过可学习的局部网格变形技术，HeadEvolver框架可以通过文本引导生成高质量的头部头像，保留细节并支持编辑和动画。 |

# 详细

[^1]: HeadEvolver：通过本地可学习网格变形实现文本到头部头像的转换

    HeadEvolver: Text to Head Avatars via Locally Learnable Mesh Deformation

    [https://arxiv.org/abs/2403.09326](https://arxiv.org/abs/2403.09326)

    通过可学习的局部网格变形技术，HeadEvolver框架可以通过文本引导生成高质量的头部头像，保留细节并支持编辑和动画。

    

    我们提出了HeadEvolver，一个新颖的框架，可以通过文本引导生成风格化的头部头像。HeadEvolver使用模板头部网格的本地可学习网格变形，生成高质量的数字资产，以实现保留细节的编辑和动画。为了解决全局变形中缺乏细粒度和语义感知本地形状控制的挑战，我们引入了可训练参数作为每个三角形的Jacobi矩阵的加权因子，以自适应地改变本地形状同时保持全局对应和面部特征。此外，为了确保来自不同视角的结果形状和外观的连贯性，我们使用预训练的图像扩散模型进行可微分渲染，并添加正则化项以在文本引导下优化变形。大量实验证明，我们的方法可以生成具有关节网格的多样化头部头像，可无缝编辑。

    arXiv:2403.09326v1 Announce Type: cross  Abstract: We present HeadEvolver, a novel framework to generate stylized head avatars from text guidance. HeadEvolver uses locally learnable mesh deformation from a template head mesh, producing high-quality digital assets for detail-preserving editing and animation. To tackle the challenges of lacking fine-grained and semantic-aware local shape control in global deformation through Jacobians, we introduce a trainable parameter as a weighting factor for the Jacobian at each triangle to adaptively change local shapes while maintaining global correspondences and facial features. Moreover, to ensure the coherence of the resulting shape and appearance from different viewpoints, we use pretrained image diffusion models for differentiable rendering with regularization terms to refine the deformation under text guidance. Extensive experiments demonstrate that our method can generate diverse head avatars with an articulated mesh that can be edited seaml
    

