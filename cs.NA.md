# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Training for Physics-Informed Neural Networks.](http://arxiv.org/abs/2310.11789) | 这篇论文提出了一种名为AT-PINNs的对抗训练策略，通过对抗样本的微调来增强物理信息神经网络（PINNs）的鲁棒性，并且可以进行具有时间因果关系的推断。 |
| [^2] | [High-dimensional multi-view clustering methods.](http://arxiv.org/abs/2303.08582) | 本论文比较了两类高维多视角聚类方法（基于图和基于子空间），重点关注了如何处理高阶相关性，并在基准数据集上进行了实验研究。 |

# 详细

[^1]: 物理信息神经网络的对抗训练

    Adversarial Training for Physics-Informed Neural Networks. (arXiv:2310.11789v1 [cs.LG])

    [http://arxiv.org/abs/2310.11789](http://arxiv.org/abs/2310.11789)

    这篇论文提出了一种名为AT-PINNs的对抗训练策略，通过对抗样本的微调来增强物理信息神经网络（PINNs）的鲁棒性，并且可以进行具有时间因果关系的推断。

    

    物理信息神经网络在解决偏微分方程问题上显示出巨大的潜力。然而，由于不足的鲁棒性，普通的PINNs在解决涉及多尺度行为或具有尖锐或振荡特征的复杂PDE时经常面临挑战。为了解决这些问题，我们基于投影梯度下降对抗攻击提出了一种对抗训练策略，被称为AT-PINNs。AT-PINNs通过对抗样本的微调来增强PINNs的鲁棒性，可以准确识别模型失效位置并在训练过程中引导模型专注于这些区域。AT-PINNs还可以通过选择围绕时间初始值的初始拟合点来进行因果推断。我们将AT-PINNs应用于具有多尺度系数的椭圆方程、具有多峰解的泊松方程、具有尖锐解的Burgers方程以及Allen-Cahn方程。

    Physics-informed neural networks have shown great promise in solving partial differential equations. However, due to insufficient robustness, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To address these issues, based on the projected gradient descent adversarial attack, we proposed an adversarial training strategy for PINNs termed by AT-PINNs. AT-PINNs enhance the robustness of PINNs by fine-tuning the model with adversarial samples, which can accurately identify model failure locations and drive the model to focus on those regions during training. AT-PINNs can also perform inference with temporal causality by selecting the initial collocation points around temporal initial values. We implement AT-PINNs to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, Burgers equation with sharp solutions and the Allen-Cahn equati
    
[^2]: 高维多视角聚类方法

    High-dimensional multi-view clustering methods. (arXiv:2303.08582v1 [cs.LG])

    [http://arxiv.org/abs/2303.08582](http://arxiv.org/abs/2303.08582)

    本论文比较了两类高维多视角聚类方法（基于图和基于子空间），重点关注了如何处理高阶相关性，并在基准数据集上进行了实验研究。

    

    最近几年，相比于单视角聚类，多视角聚类被广泛应用于数据分析中。它可以提供更多的数据信息，但也带来了一些挑战，如如何组合这些视角或特征。最近的研究主要集中在张量表示上，而不是将数据视为简单的矩阵。这种方法可以处理数据之间的高阶相关性，而基于矩阵的方法则难以捕捉这种相关性。因此，我们将研究和比较这些方法，特别是基于图的聚类和子空间聚类，以及在基准数据集上的实验结果。

    Multi-view clustering has been widely used in recent years in comparison to single-view clustering, for clear reasons, as it offers more insights into the data, which has brought with it some challenges, such as how to combine these views or features. Most of recent work in this field focuses mainly on tensor representation instead of treating the data as simple matrices. This permits to deal with the high-order correlation between the data which the based matrix approach struggles to capture. Accordingly, we will examine and compare these approaches, particularly in two categories, namely graph-based clustering and subspace-based clustering. We will conduct and report experiments of the main clustering methods over a benchmark datasets.
    

