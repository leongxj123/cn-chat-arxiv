# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bounding Reconstruction Attack Success of Adversaries Without Data Priors](https://arxiv.org/abs/2402.12861) | 本研究提供了差分隐私训练的机器学习模型在现实对抗设置下重建成功率的正式上限，并通过实证结果支持，有助于更明智地选择隐私参数。 |
| [^2] | [Unlearnable Algorithms for In-context Learning](https://arxiv.org/abs/2402.00751) | 本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。 |

# 详细

[^1]: 在没有数据先验条件下限制对抗者重建攻击成功率

    Bounding Reconstruction Attack Success of Adversaries Without Data Priors

    [https://arxiv.org/abs/2402.12861](https://arxiv.org/abs/2402.12861)

    本研究提供了差分隐私训练的机器学习模型在现实对抗设置下重建成功率的正式上限，并通过实证结果支持，有助于更明智地选择隐私参数。

    

    机器学习模型的重建攻击存在泄漏敏感数据的风险。在特定情境下，对手可以使用模型的梯度几乎完美地重建训练数据样本。在使用差分隐私（DP）训练机器学习模型时，可以提供对这种重建攻击成功率的正式上限。迄今为止，这些上限是在可能不符合高度现实实用性的最坏情况假设下制定的。在本文中，我们针对差分隐私训练的机器学习模型提供了在现实对抗设置下的重建成功率正式上限，并通过实证结果支持这些上限。通过这一点，我们展示了在现实情境中，（a）预期的重建成功率可以在不同背景和不同度量下得到适当的限制，这（b）有助于更明智地选择隐私参数。

    arXiv:2402.12861v1 Announce Type: new  Abstract: Reconstruction attacks on machine learning (ML) models pose a strong risk of leakage of sensitive data. In specific contexts, an adversary can (almost) perfectly reconstruct training data samples from a trained model using the model's gradients. When training ML models with differential privacy (DP), formal upper bounds on the success of such reconstruction attacks can be provided. So far, these bounds have been formulated under worst-case assumptions that might not hold high realistic practicality. In this work, we provide formal upper bounds on reconstruction success under realistic adversarial settings against ML models trained with DP and support these bounds with empirical results. With this, we show that in realistic scenarios, (a) the expected reconstruction success can be bounded appropriately in different contexts and by different metrics, which (b) allows for a more educated choice of a privacy parameter.
    
[^2]: 无法学习的算法用于上下文学习

    Unlearnable Algorithms for In-context Learning

    [https://arxiv.org/abs/2402.00751](https://arxiv.org/abs/2402.00751)

    本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。

    

    随着模型被越来越多地部署在未知来源的数据上，机器去学习变得越来越受欢迎。然而，要实现精确的去学习——在没有使用要遗忘的数据的情况下获得与模型分布匹配的模型——是具有挑战性或低效的，通常需要大量的重新训练。在本文中，我们专注于预先训练的大型语言模型（LLM）的任务适应阶段的高效去学习方法。我们观察到LLM进行任务适应的上下文学习能力可以实现任务适应训练数据的高效精确去学习。我们提供了一种算法，用于选择少量训练示例加到LLM的提示前面（用于任务适应），名为ERASE，它的去学习操作成本与模型和数据集的大小无关，意味着它适用于大型模型和数据集。我们还将我们的方法与微调方法进行了比较，并讨论了两种方法之间的权衡。这使我们得到了以下结论：

    Machine unlearning is a desirable operation as models get increasingly deployed on data with unknown provenance. However, achieving exact unlearning -- obtaining a model that matches the model distribution when the data to be forgotten was never used -- is challenging or inefficient, often requiring significant retraining. In this paper, we focus on efficient unlearning methods for the task adaptation phase of a pretrained large language model (LLM). We observe that an LLM's ability to do in-context learning for task adaptation allows for efficient exact unlearning of task adaptation training data. We provide an algorithm for selecting few-shot training examples to prepend to the prompt given to an LLM (for task adaptation), ERASE, whose unlearning operation cost is independent of model and dataset size, meaning it scales to large models and datasets. We additionally compare our approach to fine-tuning approaches and discuss the trade-offs between the two approaches. This leads us to p
    

