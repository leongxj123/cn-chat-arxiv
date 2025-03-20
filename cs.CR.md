# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness Bounds on the Successful Adversarial Examples: Theory and Practice](https://arxiv.org/abs/2403.01896) | 本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。 |

# 详细

[^1]: 成功对抗样本的强鲁棒性界限：理论与实践

    Robustness Bounds on the Successful Adversarial Examples: Theory and Practice

    [https://arxiv.org/abs/2403.01896](https://arxiv.org/abs/2403.01896)

    本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。

    

    对抗样本（AE）是一种针对机器学习的攻击方法，通过对数据添加不可感知的扰动来诱使错分。本文基于高斯过程（GP）分类，研究了成功AE的概率上限。我们证明了一个新的上界，取决于AE的扰动范数、GP中使用的核函数以及训练数据集中具有不同标签的最接近对之间的距离。令人惊讶的是，该上限不受样本数据集分布的影响。我们通过使用ImageNet的实验验证了我们的理论结果。此外，我们展示了改变核函数参数会导致成功AE概率上限的变化。

    arXiv:2403.01896v1 Announce Type: new  Abstract: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.
    

