# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |
| [^2] | [Position Paper: Assessing Robustness, Privacy, and Fairness in Federated Learning Integrated with Foundation Models](https://arxiv.org/abs/2402.01857) | 本文评估了基于Foundation模型集成联邦学习中鲁棒性、隐私和公平性的挑战和问题，并提出了应对策略和研究方向。 |

# 详细

[^1]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    
[^2]: 评估基于Foundation模型集成联邦学习的鲁棒性、隐私和公平性的立场论文

    Position Paper: Assessing Robustness, Privacy, and Fairness in Federated Learning Integrated with Foundation Models

    [https://arxiv.org/abs/2402.01857](https://arxiv.org/abs/2402.01857)

    本文评估了基于Foundation模型集成联邦学习中鲁棒性、隐私和公平性的挑战和问题，并提出了应对策略和研究方向。

    

    联邦学习（FL）是分散式机器学习的重大突破，但面临诸多挑战，如数据可用性有限和计算资源的变化性，这可能会限制模型的性能和可伸缩性。将Foundation模型（FM）集成到FL中，可以解决这些问题，通过预训练和数据增强增加数据丰富性并减少计算需求。然而，这种集成引入了鲁棒性、隐私和公平性方面的新问题，在现有研究中尚未得到充分解决。我们通过系统评估FM-FL集成对这些方面的影响，进行了初步调查。我们分析了其中的权衡取舍，揭示了该集成引入的威胁和问题，并提出了一套用于应对这些挑战的标准和策略。此外，我们还鉴定了可能解决这些问题的一些前景方向和研究方向。

    Federated Learning (FL), while a breakthrough in decentralized machine learning, contends with significant challenges such as limited data availability and the variability of computational resources, which can stifle the performance and scalability of the models. The integration of Foundation Models (FMs) into FL presents a compelling solution to these issues, with the potential to enhance data richness and reduce computational demands through pre-training and data augmentation. However, this incorporation introduces novel issues in terms of robustness, privacy, and fairness, which have not been sufficiently addressed in the existing research. We make a preliminary investigation into this field by systematically evaluating the implications of FM-FL integration across these dimensions. We analyze the trade-offs involved, uncover the threats and issues introduced by this integration, and propose a set of criteria and strategies for navigating these challenges. Furthermore, we identify po
    

