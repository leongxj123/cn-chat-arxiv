# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning truly monotone operators with applications to nonlinear inverse problems](https://arxiv.org/abs/2404.00390) | 通过新定义的惩罚损失学习单调神经网络，解决图像处理中的问题，并利用FBF算法提供收敛保证，以解决非线性逆问题。 |

# 详细

[^1]: 学习真正单调算子及其在非线性逆问题中的应用

    Learning truly monotone operators with applications to nonlinear inverse problems

    [https://arxiv.org/abs/2404.00390](https://arxiv.org/abs/2404.00390)

    通过新定义的惩罚损失学习单调神经网络，解决图像处理中的问题，并利用FBF算法提供收敛保证，以解决非线性逆问题。

    

    本文介绍了一种通过新定义的惩罚损失来学习单调神经网络的新方法。该方法在解决一类变分问题中特别有效，特别是图像处理任务中常遇到的单调包含问题。采用前-后-前（FBF）算法来解决这些问题，在神经网络的Lipschitz常数未知的情况下也能提供解决方案。值得注意的是，FBF算法在学习算子单调的条件下提供收敛保证。借鉴即插即用的方法，我们的目标是将这些新学习的算子应用于解决非线性逆问题。为实现这一目标，我们首先将问题制定为一个变分包含问题，随后训练一个单调神经网络来逼近一个本质上可能不是单调的算子。利用FBF算法

    arXiv:2404.00390v1 Announce Type: cross  Abstract: This article introduces a novel approach to learning monotone neural networks through a newly defined penalization loss. The proposed method is particularly effective in solving classes of variational problems, specifically monotone inclusion problems, commonly encountered in image processing tasks. The Forward-Backward-Forward (FBF) algorithm is employed to address these problems, offering a solution even when the Lipschitz constant of the neural network is unknown. Notably, the FBF algorithm provides convergence guarantees under the condition that the learned operator is monotone. Building on plug-and-play methodologies, our objective is to apply these newly learned operators to solving non-linear inverse problems. To achieve this, we initially formulate the problem as a variational inclusion problem. Subsequently, we train a monotone neural network to approximate an operator that may not inherently be monotone. Leveraging the FBF al
    

