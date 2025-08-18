# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Data for Robust Stroke Segmentation](https://arxiv.org/abs/2404.01946) | 提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。 |
| [^2] | [JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example.](http://arxiv.org/abs/2401.01199) | JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。 |

# 详细

[^1]: 用于鲁棒性中风分割的合成数据

    Synthetic Data for Robust Stroke Segmentation

    [https://arxiv.org/abs/2404.01946](https://arxiv.org/abs/2404.01946)

    提出一种用于中风分割的合成框架，使用病变特定增强策略扩展了SynthSeg方法，通过训练深度学习模型实现对健康组织和病理病变的分割，无需特定序列的训练数据，在领域内和领域外数据集的评估中表现出鲁棒性能。

    

    arXiv:2404.01946v1 公告类型：交叉 摘要：目前基于深度学习的神经影像语义分割需要高分辨率扫描和大量注释数据集，这给临床适用性带来了显著障碍。我们提出了一种新颖的合成框架，用于病变分割任务，扩展了已建立的SynthSeg方法的能力，以适应具有病变特定增强策略的大型异质病变。我们的方法使用从健康和中风数据集派生的标签映射训练深度学习模型，在这里演示了UNet架构，促进了健康组织和病理病变的分割，而无需特定于序列的训练数据。针对领域内和领域外（OOD）数据集进行评估，我们的框架表现出鲁棒性能，与训练领域内的当前方法相媲美，并在OOD数据上显着优于它们。这一贡献有望推动医学...

    arXiv:2404.01946v1 Announce Type: cross  Abstract: Deep learning-based semantic segmentation in neuroimaging currently requires high-resolution scans and extensive annotated datasets, posing significant barriers to clinical applicability. We present a novel synthetic framework for the task of lesion segmentation, extending the capabilities of the established SynthSeg approach to accommodate large heterogeneous pathologies with lesion-specific augmentation strategies. Our method trains deep learning models, demonstrated here with the UNet architecture, using label maps derived from healthy and stroke datasets, facilitating the segmentation of both healthy tissue and pathological lesions without sequence-specific training data. Evaluated against in-domain and out-of-domain (OOD) datasets, our framework demonstrates robust performance, rivaling current methods within the training domain and significantly outperforming them on OOD data. This contribution holds promise for advancing medical
    
[^2]: JMA:一种快速生成几乎最优定向对抗样本的通用算法

    JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example. (arXiv:2401.01199v1 [cs.LG])

    [http://arxiv.org/abs/2401.01199](http://arxiv.org/abs/2401.01199)

    JMA是一种通用算法，用于生成几乎最优的定向对抗样本。该算法通过最小化Jacobian引起的马氏距离，考虑了将输入样本的潜在空间表示在给定方向上移动所需的投入。该算法在解决对抗样本问题方面提供了最优解。

    

    目前为止，大多数用于生成针对深度学习分类器的定向对抗样本的方法都是高度次优的，通常依赖于增加目标类别的可能性，因此隐含地专注于一热编码设置。在本文中，我们提出了一种更加通用的、理论上可靠的定向攻击方法，该方法利用最小化雅可比引起的马氏距离（JMA）项，考虑将输入样本的潜在空间表示在给定方向上移动所需的投入（在输入空间中）。通过利用沃尔夫二重性定理求解最小化问题，将问题简化为解非负最小二乘（NNLS）问题。所提出的算法为Szegedy等人最初引入的对抗样本问题的线性化版本提供了最优解。我们进行的实验证实了所提出的攻击的广泛性。

    Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be 
    

