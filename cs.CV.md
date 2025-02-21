# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It](https://arxiv.org/abs/2403.14715) | LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。 |
| [^2] | [On Memorization in Diffusion Models.](http://arxiv.org/abs/2310.02664) | 本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。 |

# 详细

[^1]: 理解为何标签平滑会降低选择性分类的效果以及如何解决这个问题

    Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It

    [https://arxiv.org/abs/2403.14715](https://arxiv.org/abs/2403.14715)

    LS方法在深度神经网络分类器训练中的标签平滑效果被发现会负面影响选择性分类，通过影响模型预测不确定性，此研究阐明了这一现象。

    

    标签平滑（LS）是一种流行的深度神经网络分类器训练的正则化方法，因为它在提高测试准确性方面效果显著，并且实现简单。"硬"的one-hot标签通过将概率质量均匀分配给其他类别来进行"平滑化"，从而减少过度拟合。在这项工作中，我们揭示了LS如何负面影响选择性分类（SC）- 其目标是利用模型的预测不确定性来拒绝错误分类。我们首先在一系列任务和架构中从经验上证明LS会导致SC的一致性降级。然后，我们通过分析logit级别的梯度来解释这一点，表明LS通过在错误概率低时更加正则化最大logit，而在错误概率高时更少正则化，加剧了过度自信和低自信。这阐明了以前报道的强分类器在SC中性能不佳的实验结果。

    arXiv:2403.14715v1 Announce Type: cross  Abstract: Label smoothing (LS) is a popular regularisation method for training deep neural network classifiers due to its effectiveness in improving test accuracy and its simplicity in implementation. "Hard" one-hot labels are "smoothed" by uniformly distributing probability mass to other classes, reducing overfitting. In this work, we reveal that LS negatively affects selective classification (SC) - where the aim is to reject misclassifications using a model's predictive uncertainty. We first demonstrate empirically across a range of tasks and architectures that LS leads to a consistent degradation in SC. We then explain this by analysing logit-level gradients, showing that LS exacerbates overconfidence and underconfidence by regularising the max logit more when the probability of error is low, and less when the probability of error is high. This elucidates previously reported experimental results where strong classifiers underperform in SC. We
    
[^2]: 关于扩散模型记忆化的研究

    On Memorization in Diffusion Models. (arXiv:2310.02664v1 [cs.LG])

    [http://arxiv.org/abs/2310.02664](http://arxiv.org/abs/2310.02664)

    本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。

    

    近年来，由于其生成新颖高质量样本的能力，扩散模型引起了广泛的研究兴趣。然而，通过典型的训练目标，即去噪得分匹配，扩散模型只能生成复制训练数据的样本，这表明在理论上会出现记忆化的行为，这与现有先进扩散模型的普遍泛化能力相矛盾，因此需要深入理解。我们观察到记忆化行为倾向于在较小的数据集上发生，我们提出了有效模型记忆化(EMM)的定义，这是一种衡量学习的扩散模型在最大数据集上近似其理论最优点的度量标准。然后，我们量化了影响这些记忆化行为的重要因素，重点关注数据分布和模型配置。

    Due to their capacity to generate novel and high-quality samples, diffusion models have attracted significant research interest in recent years. Notably, the typical training objective of diffusion models, i.e., denoising score matching, has a closed-form optimal solution that can only generate training data replicating samples. This indicates that a memorization behavior is theoretically expected, which contradicts the common generalization ability of state-of-the-art diffusion models, and thus calls for a deeper understanding. Looking into this, we first observe that memorization behaviors tend to occur on smaller-sized datasets, which motivates our definition of effective model memorization (EMM), a metric measuring the maximum size of training data at which a learned diffusion model approximates its theoretical optimum. Then, we quantify the impact of the influential factors on these memorization behaviors in terms of EMM, focusing primarily on data distribution, model configuratio
    

