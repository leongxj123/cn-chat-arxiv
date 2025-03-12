# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [INPC: Implicit Neural Point Clouds for Radiance Field Rendering](https://arxiv.org/abs/2403.16862) | 提出了一种新颖的隐式点云表示方法，结合了连续八叉树概率场和多分辨率哈希网格，实现了快速渲染和保留细致几何细节的优势，并且在几个常见基准数据集上实现了最先进的图像质量。 |
| [^2] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^3] | [Structure Preserving Diffusion Models](https://arxiv.org/abs/2402.19369) | 提出了一种结构保持的扩散过程，可以学习具有群对称性等额外结构的分布，并开发了一系列对称等变扩散模型来实现这一点。 |
| [^4] | [Associative Transformer Is A Sparse Representation Learner.](http://arxiv.org/abs/2309.12862) | 关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。 |

# 详细

[^1]: INPC：用于辐射场渲染的隐式神经点云

    INPC: Implicit Neural Point Clouds for Radiance Field Rendering

    [https://arxiv.org/abs/2403.16862](https://arxiv.org/abs/2403.16862)

    提出了一种新颖的隐式点云表示方法，结合了连续八叉树概率场和多分辨率哈希网格，实现了快速渲染和保留细致几何细节的优势，并且在几个常见基准数据集上实现了最先进的图像质量。

    

    我们引入了一种新的方法，用于重建和合成无边界的现实世界场景。与以往使用体积场、基于网格的模型或离散点云代理的方法相比，我们提出了一种混合场景表示，它在连续八叉树概率场和多分辨率哈希网格中隐含地编码点云。通过这样做，我们结合了两个世界的优势，保留了在优化过程中有利的行为：我们的新颖隐式点云表示和可微的双线性光栅化器实现了快速渲染，同时保留了细微的几何细节，而无需依赖于像结构运动点云这样的初始先验。我们的方法在几个常见基准数据集上实现了最先进的图像质量。此外，我们实现了快速推理，可交互帧速率，并且可以提取显式点云以进一步提高性能。

    arXiv:2403.16862v1 Announce Type: cross  Abstract: We introduce a new approach for reconstruction and novel-view synthesis of unbounded real-world scenes. In contrast to previous methods using either volumetric fields, grid-based models, or discrete point cloud proxies, we propose a hybrid scene representation, which implicitly encodes a point cloud in a continuous octree-based probability field and a multi-resolution hash grid. In doing so, we combine the benefits of both worlds by retaining favorable behavior during optimization: Our novel implicit point cloud representation and differentiable bilinear rasterizer enable fast rendering while preserving fine geometric detail without depending on initial priors like structure-from-motion point clouds. Our method achieves state-of-the-art image quality on several common benchmark datasets. Furthermore, we achieve fast inference at interactive frame rates, and can extract explicit point clouds to further enhance performance.
    
[^2]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^3]: 结构保持的扩散模型

    Structure Preserving Diffusion Models

    [https://arxiv.org/abs/2402.19369](https://arxiv.org/abs/2402.19369)

    提出了一种结构保持的扩散过程，可以学习具有群对称性等额外结构的分布，并开发了一系列对称等变扩散模型来实现这一点。

    

    近年来，扩散模型已成为主要的分布学习方法。在本文中，我们介绍了结构保持的扩散过程，这是一类用于学习具有额外结构（如群对称性）的分布的扩散过程，通过制定扩散转换步骤保持对称性的理论条件。除了实现等变数据采样轨迹外，我们通过开发一系列不同的对称等变扩散模型来说明这些结果，这些模型能够学习固有对称的分布。我们使用实证研究验证所开发的模型符合提出的理论，并在样本均等性方面能够胜过现有方法。我们还展示了如何利用提出的模型实现理论上保证的等变图像噪声。

    arXiv:2402.19369v1 Announce Type: new  Abstract: Diffusion models have become the leading distribution-learning method in recent years. Herein, we introduce structure-preserving diffusion processes, a family of diffusion processes for learning distributions that possess additional structure, such as group symmetries, by developing theoretical conditions under which the diffusion transition steps preserve said symmetry. While also enabling equivariant data sampling trajectories, we exemplify these results by developing a collection of different symmetry equivariant diffusion models capable of learning distributions that are inherently symmetric. Empirical studies, over both synthetic and real-world datasets, are used to validate the developed models adhere to the proposed theory and are capable of achieving improved performance over existing methods in terms of sample equality. We also show how the proposed models can be used to achieve theoretically guaranteed equivariant image noise r
    
[^4]: 关联变换器是一种稀疏表示学习器

    Associative Transformer Is A Sparse Representation Learner. (arXiv:2309.12862v1 [cs.LG])

    [http://arxiv.org/abs/2309.12862](http://arxiv.org/abs/2309.12862)

    关联变换器（AiT）是一种采用低秩显式记忆和关联记忆的稀疏表示学习器，通过联合端到端训练实现模块特化和注意力瓶颈的形成。

    

    在传统的Transformer模型中，出现了一种新兴的基于稀疏交互的注意力机制，这种机制与生物原理更为接近。包括Set Transformer和Perceiver在内的方法采用了与有限能力的潜在空间相结合的交叉注意力机制。基于最近对全局工作空间理论和关联记忆的神经科学研究，我们提出了关联变换器（AiT）。AiT引入了低秩显式记忆，既可以作为先验来指导共享工作空间的瓶颈注意力，又可以作为关联记忆的吸引子。通过联合端到端训练，这些先验自然地发展出模块的特化，每个模块对形成注意力瓶颈的归纳偏好有所贡献。瓶颈可以促进输入之间为将信息写入内存而进行竞争。我们展示了AiT是一种稀疏表示学习器。

    Emerging from the monolithic pairwise attention mechanism in conventional Transformer models, there is a growing interest in leveraging sparse interactions that align more closely with biological principles. Approaches including the Set Transformer and the Perceiver employ cross-attention consolidated with a latent space that forms an attention bottleneck with limited capacity. Building upon recent neuroscience studies of Global Workspace Theory and associative memory, we propose the Associative Transformer (AiT). AiT induces low-rank explicit memory that serves as both priors to guide bottleneck attention in the shared workspace and attractors within associative memory of a Hopfield network. Through joint end-to-end training, these priors naturally develop module specialization, each contributing a distinct inductive bias to form attention bottlenecks. A bottleneck can foster competition among inputs for writing information into the memory. We show that AiT is a sparse representation 
    

