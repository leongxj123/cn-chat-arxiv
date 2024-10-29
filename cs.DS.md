# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Differentially Private Subspace Estimation Without Distributional Assumptions](https://arxiv.org/abs/2402.06465) | 本论文研究了在没有分布假设的情况下，差分隐私子空间估计的问题。通过使用少量的数据点，可以私密地识别出低维结构，避免了高维度的代价。 |

# 详细

[^1]: 关于无分布假设的差分隐私子空间估计

    On Differentially Private Subspace Estimation Without Distributional Assumptions

    [https://arxiv.org/abs/2402.06465](https://arxiv.org/abs/2402.06465)

    本论文研究了在没有分布假设的情况下，差分隐私子空间估计的问题。通过使用少量的数据点，可以私密地识别出低维结构，避免了高维度的代价。

    

    隐私数据分析面临着一个被称为维数诅咒的重大挑战，导致了成本的增加。然而，许多数据集具有固有的低维结构。例如，在梯度下降优化过程中，梯度经常位于一个低维子空间附近。如果可以使用少量点私密地识别出这种低维结构，就可以避免因高维度而支付隐私和准确性的代价。

    Private data analysis faces a significant challenge known as the curse of dimensionality, leading to increased costs. However, many datasets possess an inherent low-dimensional structure. For instance, during optimization via gradient descent, the gradients frequently reside near a low-dimensional subspace. If the low-dimensional structure could be privately identified using a small amount of points, we could avoid paying (in terms of privacy and accuracy) for the high ambient dimension.   On the negative side, Dwork, Talwar, Thakurta, and Zhang (STOC 2014) proved that privately estimating subspaces, in general, requires an amount of points that depends on the dimension. But Singhal and Steinke (NeurIPS 2021) bypassed this limitation by considering points that are i.i.d. samples from a Gaussian distribution whose covariance matrix has a certain eigenvalue gap. Yet, it was still left unclear whether we could provide similar upper bounds without distributional assumptions and whether we 
    

