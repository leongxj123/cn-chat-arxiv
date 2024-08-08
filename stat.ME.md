# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric Linear Feature Learning in Regression Through Regularisation.](http://arxiv.org/abs/2307.12754) | 本研究提出了一种新的非参数线性特征学习方法，对于监督学习中存在于低维线性子空间中的相关信息的预测和解释能力的提升是非常有帮助的。 |

# 详细

[^1]: 非参数线性特征学习在回归中的应用通过正则化

    Nonparametric Linear Feature Learning in Regression Through Regularisation. (arXiv:2307.12754v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2307.12754](http://arxiv.org/abs/2307.12754)

    本研究提出了一种新的非参数线性特征学习方法，对于监督学习中存在于低维线性子空间中的相关信息的预测和解释能力的提升是非常有帮助的。

    

    表征学习在自动化特征选择中发挥着关键作用，特别是在高维数据的背景下，非参数方法常常很难应对。在本研究中，我们专注于监督学习场景，其中相关信息存在于数据的低维线性子空间中，即多指数模型。如果已知该子空间，将大大增强预测、计算和解释能力。为了解决这一挑战，我们提出了一种新颖的非参数预测的线性特征学习方法，同时估计预测函数和线性子空间。我们的方法采用经验风险最小化，并加上函数导数的惩罚项，以保证其多样性。通过利用Hermite多项式的正交性和旋转不变性特性，我们引入了我们的估计器RegFeaL。通过利用替代最小化，我们迭代地旋转数据以改善与线性子空间的对齐。

    Representation learning plays a crucial role in automated feature selection, particularly in the context of high-dimensional data, where non-parametric methods often struggle. In this study, we focus on supervised learning scenarios where the pertinent information resides within a lower-dimensional linear subspace of the data, namely the multi-index model. If this subspace were known, it would greatly enhance prediction, computation, and interpretation. To address this challenge, we propose a novel method for linear feature learning with non-parametric prediction, which simultaneously estimates the prediction function and the linear subspace. Our approach employs empirical risk minimisation, augmented with a penalty on function derivatives, ensuring versatility. Leveraging the orthogonality and rotation invariance properties of Hermite polynomials, we introduce our estimator, named RegFeaL. By utilising alternative minimisation, we iteratively rotate the data to improve alignment with 
    

