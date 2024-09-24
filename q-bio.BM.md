# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Symphony: Symmetry-Equivariant Point-Centered Spherical Harmonics for Molecule Generation](https://arxiv.org/abs/2311.16199) | Symphony提出了一种新颖的$E(3)$-等变自回归生成模型，通过使用点对称球形谐波信号来高效建模分子的3D几何结构。 |

# 详细

[^1]: Symphony: 对分子生成的点对称球形谐波的对称等变自回归模型

    Symphony: Symmetry-Equivariant Point-Centered Spherical Harmonics for Molecule Generation

    [https://arxiv.org/abs/2311.16199](https://arxiv.org/abs/2311.16199)

    Symphony提出了一种新颖的$E(3)$-等变自回归生成模型，通过使用点对称球形谐波信号来高效建模分子的3D几何结构。

    

    我们提出了Symphony，这是一个$E(3)$-等变的自回归生成模型，用于3D分子几何结构的构建，通过从分子碎片中迭代地构建分子。现有的自回归模型如G-SchNet和G-SphereNet用于分子的旋转不变特征来尊重分子的3D对称性。相反，Symphony使用带有更高次$E(3)$-等变特征的消息传递。这使得通过球谐信号有效地建模分子的3D几何的概率分布成为可能。我们展示了Symphony能够准确地从QM9数据集中生成小分子，优于现有的自回归模型，并接近扩散模型的性能。

    arXiv:2311.16199v2 Announce Type: replace  Abstract: We present Symphony, an $E(3)$-equivariant autoregressive generative model for 3D molecular geometries that iteratively builds a molecule from molecular fragments. Existing autoregressive models such as G-SchNet and G-SphereNet for molecules utilize rotationally invariant features to respect the 3D symmetries of molecules. In contrast, Symphony uses message-passing with higher-degree $E(3)$-equivariant features. This allows a novel representation of probability distributions via spherical harmonic signals to efficiently model the 3D geometry of molecules. We show that Symphony is able to accurately generate small molecules from the QM9 dataset, outperforming existing autoregressive models and approaching the performance of diffusion models.
    

