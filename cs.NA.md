# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Physics Informed Neural Networks.](http://arxiv.org/abs/2401.02300) | 引入了一种鲁棒的物理信息神经网络（RPINNs）来近似偏微分方程（PDE）的解，该网络在训练过程中考虑了PDE的控制物理法则，解决了传统PINNs中损失函数与真实误差不鲁棒的问题。 |

# 详细

[^1]: 鲁棒的物理信息神经网络

    Robust Physics Informed Neural Networks. (arXiv:2401.02300v1 [cs.LG])

    [http://arxiv.org/abs/2401.02300](http://arxiv.org/abs/2401.02300)

    引入了一种鲁棒的物理信息神经网络（RPINNs）来近似偏微分方程（PDE）的解，该网络在训练过程中考虑了PDE的控制物理法则，解决了传统PINNs中损失函数与真实误差不鲁棒的问题。

    

    我们引入了一种鲁棒版本的物理信息神经网络（RPINNs）来近似偏微分方程（PDE）的解。标准的物理信息神经网络（PINN）在学习过程中考虑了由PDE描述的控制物理法则。该网络在由物理域和边界随机选择的数据集上进行训练。PINNs已成功应用于解决由PDE和边界条件描述的各种问题。传统PINNs中的损失函数基于PDE的强残差。这种PINNs中的损失函数通常对真实误差不具有鲁棒性。PINNs中的损失函数与真实误差可能相差很大，这使得训练过程更加困难。特别是，如果我们不知道精确解，我们就不能估计训练过程是否已经以所需的精度收敛到解。这在我们不知道精确解时尤其正确，

    We introduce a Robust version of the Physics-Informed Neural Networks (RPINNs) to approximate the Partial Differential Equations (PDEs) solution. Standard Physics Informed Neural Networks (PINN) takes into account the governing physical laws described by PDE during the learning process. The network is trained on a data set that consists of randomly selected points in the physical domain and its boundary. PINNs have been successfully applied to solve various problems described by PDEs with boundary conditions. The loss function in traditional PINNs is based on the strong residuals of the PDEs. This loss function in PINNs is generally not robust with respect to the true error. The loss function in PINNs can be far from the true error, which makes the training process more difficult. In particular, we do not know if the training process has already converged to the solution with the required accuracy. This is especially true if we do not know the exact solution, so we cannot estimate the 
    

