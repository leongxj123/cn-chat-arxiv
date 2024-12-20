# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Discretized Neural Networks under Ricci Flow.](http://arxiv.org/abs/2302.03390) | 本文使用信息几何构造了线性几乎欧几里得流形，通过引入偏微分方程Ricci流，解决了离散神经网络训练中梯度无穷或零的问题。 |

# 详细

[^1]: 在Ricci流下学习离散神经网络

    Learning Discretized Neural Networks under Ricci Flow. (arXiv:2302.03390v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03390](http://arxiv.org/abs/2302.03390)

    本文使用信息几何构造了线性几乎欧几里得流形，通过引入偏微分方程Ricci流，解决了离散神经网络训练中梯度无穷或零的问题。

    

    本文考虑了由低精度权重和激活函数构成的离散神经网络（DNNs）在训练过程中由于非可微分离散函数而遭受无穷或零梯度的问题。本文针对此问题，提出了把 STE近似梯度看作整体偏差的度量扰动，通过对偶理论将其看作黎曼流形上的度量扰动，并在信息几何的基础上为 DNN构造了线性几乎欧几里得（LNE）流形以处理扰动。

    In this paper, we consider Discretized Neural Networks (DNNs) consisting of low-precision weights and activations, which suffer from either infinite or zero gradients due to the non-differentiable discrete function in the training process. In this case, most training-based DNNs employ the standard Straight-Through Estimator (STE) to approximate the gradient w.r.t. discrete values. However, the STE gives rise to the problem of gradient mismatch, due to the perturbations of the approximated gradient. To address this problem, this paper reveals that this mismatch can be viewed as a metric perturbation in a Riemannian manifold through the lens of duality theory. Further, on the basis of the information geometry, we construct the Linearly Nearly Euclidean (LNE) manifold for DNNs as a background to deal with perturbations. By introducing a partial differential equation on metrics, i.e., the Ricci flow, we prove the dynamical stability and convergence of the LNE metric with the $L^2$-norm per
    

