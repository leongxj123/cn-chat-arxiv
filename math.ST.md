# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data.](http://arxiv.org/abs/2202.05928) | 本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。 |

# 详细

[^1]: 不需要线性关系的良性过拟合：通过梯度下降训练的神经网络分类器用于噪声线性数据

    Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data. (arXiv:2202.05928v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.05928](http://arxiv.org/abs/2202.05928)

    本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。

    

    良性过拟合是指插值模型在存在噪声数据的情况下能够很好地泛化的现象，最早出现在使用梯度下降训练的神经网络模型中。为了更好地理解这一实证观察，我们考虑了两层神经网络在随机初始化后通过梯度下降在逻辑损失函数上进行插值训练的泛化误差。我们假设数据来自于明显分离的类条件对数凹分布，并允许训练标签中的一定比例被对手篡改。我们证明在这种情况下，神经网络表现出良性过拟合的特点：它们可以被驱动到零训练误差，完美地拟合任何有噪声的训练标签，并同时达到极小化最大化最优测试误差。与之前关于良性过拟合需要线性或基于核的预测器的工作相比，我们的分析在模型和学习动态都是基本非线性的情况下成立。

    Benign overfitting, the phenomenon where interpolating models generalize well in the presence of noisy data, was first observed in neural network models trained with gradient descent. To better understand this empirical observation, we consider the generalization error of two-layer neural networks trained to interpolation by gradient descent on the logistic loss following random initialization. We assume the data comes from well-separated class-conditional log-concave distributions and allow for a constant fraction of the training labels to be corrupted by an adversary. We show that in this setting, neural networks exhibit benign overfitting: they can be driven to zero training error, perfectly fitting any noisy training labels, and simultaneously achieve minimax optimal test error. In contrast to previous work on benign overfitting that require linear or kernel-based predictors, our analysis holds in a setting where both the model and learning dynamics are fundamentally nonlinear.
    

