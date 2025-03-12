# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [INPC: Implicit Neural Point Clouds for Radiance Field Rendering](https://arxiv.org/abs/2403.16862) | 提出了一种新颖的隐式点云表示方法，结合了连续八叉树概率场和多分辨率哈希网格，实现了快速渲染和保留细致几何细节的优势，并且在几个常见基准数据集上实现了最先进的图像质量。 |

# 详细

[^1]: INPC：用于辐射场渲染的隐式神经点云

    INPC: Implicit Neural Point Clouds for Radiance Field Rendering

    [https://arxiv.org/abs/2403.16862](https://arxiv.org/abs/2403.16862)

    提出了一种新颖的隐式点云表示方法，结合了连续八叉树概率场和多分辨率哈希网格，实现了快速渲染和保留细致几何细节的优势，并且在几个常见基准数据集上实现了最先进的图像质量。

    

    我们引入了一种新的方法，用于重建和合成无边界的现实世界场景。与以往使用体积场、基于网格的模型或离散点云代理的方法相比，我们提出了一种混合场景表示，它在连续八叉树概率场和多分辨率哈希网格中隐含地编码点云。通过这样做，我们结合了两个世界的优势，保留了在优化过程中有利的行为：我们的新颖隐式点云表示和可微的双线性光栅化器实现了快速渲染，同时保留了细微的几何细节，而无需依赖于像结构运动点云这样的初始先验。我们的方法在几个常见基准数据集上实现了最先进的图像质量。此外，我们实现了快速推理，可交互帧速率，并且可以提取显式点云以进一步提高性能。

    arXiv:2403.16862v1 Announce Type: cross  Abstract: We introduce a new approach for reconstruction and novel-view synthesis of unbounded real-world scenes. In contrast to previous methods using either volumetric fields, grid-based models, or discrete point cloud proxies, we propose a hybrid scene representation, which implicitly encodes a point cloud in a continuous octree-based probability field and a multi-resolution hash grid. In doing so, we combine the benefits of both worlds by retaining favorable behavior during optimization: Our novel implicit point cloud representation and differentiable bilinear rasterizer enable fast rendering while preserving fine geometric detail without depending on initial priors like structure-from-motion point clouds. Our method achieves state-of-the-art image quality on several common benchmark datasets. Furthermore, we achieve fast inference at interactive frame rates, and can extract explicit point clouds to further enhance performance.
    

