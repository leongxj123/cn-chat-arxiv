# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SOFARI: High-Dimensional Manifold-Based Inference.](http://arxiv.org/abs/2309.15032) | 本研究提出了一种基于高维流形的SOFAR推断（SOFARI）方法，通过结合Neyman近正交推断和SVD约束的Stiefel流形结构，实现了对多任务学习中潜在因子矩阵的准确推断。 |

# 详细

[^1]: SOFARI:基于高维流形的推断

    SOFARI: High-Dimensional Manifold-Based Inference. (arXiv:2309.15032v1 [stat.ME])

    [http://arxiv.org/abs/2309.15032](http://arxiv.org/abs/2309.15032)

    本研究提出了一种基于高维流形的SOFAR推断（SOFARI）方法，通过结合Neyman近正交推断和SVD约束的Stiefel流形结构，实现了对多任务学习中潜在因子矩阵的准确推断。

    

    多任务学习是一种广泛使用的技术，用于从各种任务中提取信息。最近，基于系数矩阵中的稀疏奇异值分解（SVD）的稀疏正交因子回归（SOFAR）框架被引入到可解释的多任务学习中，可以发现不同层次之间有意义的潜在特征-响应关联网络。然而，由于稀疏SVD约束的正交性约束，对潜在因子矩阵进行精确推断仍然具有挑战性。在本文中，我们提出了一种新颖的方法，称为基于高维流形的SOFAR推断（SOFARI），借鉴了Neyman近正交推断，并结合了SVD约束所施加的Stiefel流形结构。通过利用潜在的Stiefel流形结构，SOFARI为潜在左因子向量和奇异值提供了偏差校正的估计量。

    Multi-task learning is a widely used technique for harnessing information from various tasks. Recently, the sparse orthogonal factor regression (SOFAR) framework, based on the sparse singular value decomposition (SVD) within the coefficient matrix, was introduced for interpretable multi-task learning, enabling the discovery of meaningful latent feature-response association networks across different layers. However, conducting precise inference on the latent factor matrices has remained challenging due to orthogonality constraints inherited from the sparse SVD constraint. In this paper, we suggest a novel approach called high-dimensional manifold-based SOFAR inference (SOFARI), drawing on the Neyman near-orthogonality inference while incorporating the Stiefel manifold structure imposed by the SVD constraints. By leveraging the underlying Stiefel manifold structure, SOFARI provides bias-corrected estimators for both latent left factor vectors and singular values, for which we show to enj
    

