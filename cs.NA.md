# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hyperbolic Machine Learning Moment Closures for the BGK Equations.](http://arxiv.org/abs/2401.04783) | 这篇论文介绍了一种使用神经网络训练的双曲线闭包模型，用于BGK动力模型的Grad矩展开，以实现最高矩的梯度的精确闭合关系。 |

# 详细

[^1]: BGK方程的双曲线机器学习矩闭包

    Hyperbolic Machine Learning Moment Closures for the BGK Equations. (arXiv:2401.04783v1 [math.NA])

    [http://arxiv.org/abs/2401.04783](http://arxiv.org/abs/2401.04783)

    这篇论文介绍了一种使用神经网络训练的双曲线闭包模型，用于BGK动力模型的Grad矩展开，以实现最高矩的梯度的精确闭合关系。

    

    我们使用神经网络（NN）在BGK动力模型的矩数据上进行训练，引入了对Bhatnagar-Gross-Krook（BGK）动力模型的Grad矩展开的双曲线闭包。这个闭包是基于我们在输运封闭中导出的自由流极限的精确封闭关系而提出的。这个精确封闭关系将最高矩的梯度与四个较低矩的梯度相关联。与我们过去的工作一样，这里介绍的模型通过较低矩的梯度系数来学习最高矩的梯度。这意味着得到的双曲系统在最高矩上并非守恒。为了稳定性，神经网络的输出层被设计成强制双曲性和Galileo不变性。这确保模型能够在NN的训练窗口之外运行。与我们以前处理线性模型的辐射输运工作不同，BGK模型的非线性性要求更高级的训练。

    We introduce a hyperbolic closure for the Grad moment expansion of the Bhatnagar-Gross-Krook's (BGK) kinetic model using a neural network (NN) trained on BGK's moment data. This closure is motivated by the exact closure for the free streaming limit that we derived in our paper on closures in transport \cite{Huang2022-RTE1}. The exact closure relates the gradient of the highest moment to the gradient of four lower moments. As with our past work, the model presented here learns the gradient of the highest moment in terms of the coefficients of gradients for all lower ones. By necessity, this means that the resulting hyperbolic system is not conservative in the highest moment. For stability, the output layers of the NN are designed to enforce hyperbolicity and Galilean invariance. This ensures the model can be run outside of the training window of the NN. Unlike our previous work on radiation transport that dealt with linear models, the BGK model's nonlinearity demanded advanced training 
    

