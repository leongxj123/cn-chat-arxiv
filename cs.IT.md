# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction](https://arxiv.org/abs/2311.09386) | 本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。 |

# 详细

[^1]: 超越PCA：一种概率性Gram-Schmidt方法的特征提取

    Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction

    [https://arxiv.org/abs/2311.09386](https://arxiv.org/abs/2311.09386)

    本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。

    

    在无监督学习中，线性特征提取在数据中存在非线性依赖的情况下是一个基本挑战。我们提出使用概率性Gram-Schmidt (GS)类型的正交化过程来检测和映射出冗余维度。具体而言，通过在一族函数上应用GS过程，该族函数预计捕捉到数据中的非线性依赖性，我们构建了一系列协方差矩阵，可以用于识别新的大方差方向，或者将这些依赖性从主成分中去除。在前一种情况下，我们提供了熵减少的信息理论保证。在后一种情况下，我们证明在某些假设下，所得算法在所选择函数族的线性张成空间中可以检测和去除非线性依赖性。两种提出的方法都可以从数据中提取线性特征并去除非线性冗余。

    Linear feature extraction at the presence of nonlinear dependencies among the data is a fundamental challenge in unsupervised learning. We propose using a probabilistic Gram-Schmidt (GS) type orthogonalization process in order to detect and map out redundant dimensions. Specifically, by applying the GS process over a family of functions which presumably captures the nonlinear dependencies in the data, we construct a series of covariance matrices that can either be used to identify new large-variance directions, or to remove those dependencies from the principal components. In the former case, we provide information-theoretic guarantees in terms of entropy reduction. In the latter, we prove that under certain assumptions the resulting algorithms detect and remove nonlinear dependencies whenever those dependencies lie in the linear span of the chosen function family. Both proposed methods extract linear features from the data while removing nonlinear redundancies. We provide simulation r
    

