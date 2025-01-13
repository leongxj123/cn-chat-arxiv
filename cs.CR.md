# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial Detection by Approximation of Ensemble Boundary.](http://arxiv.org/abs/2211.10227) | 本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。 |
| [^2] | [Cyber Risk Assessment for Capital Management.](http://arxiv.org/abs/2205.08435) | 本文提出了一个两柱网络风险管理框架，其中包括网络风险评估和网络资本管理。通过基于历史网络事件数据集的案例研究，展示了全面的成本效益分析对于网络风险管理的重要性，并说明最佳策略取决于不同因素。 |

# 详细

[^1]: 使用集成边界逼近的对抗检测方法

    Adversarial Detection by Approximation of Ensemble Boundary. (arXiv:2211.10227v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.10227](http://arxiv.org/abs/2211.10227)

    本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。

    

    本论文提出了一种新的对抗攻击检测方法，针对解决两类模式识别问题的深度神经网络（DNN）集成。该集成使用Walsh系数进行组合，能够逼近布尔函数并控制集成决策边界的复杂性。本文的假设是高曲率的决策边界允许找到对抗扰动，但会改变决策边界的曲率，而与清晰图像相比，使用Walsh系数对其进行逼近的方式也有所不同。通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实验证明了攻击的可迁移性可用于检测。此外，逼近决策边界可能有助于理解DNN的学习和可迁移性特性。尽管本文的实验使用图像，所提出的方法可以用于建模两类模式识别问题的集成边界逼近。

    A new method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the complexity of the ensemble decision boundary. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it is shown experimentally that transferability of attack may be used for detection. Furthermore, approximating the decision boundary may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class en
    
[^2]: 资本管理的网络风险评估

    Cyber Risk Assessment for Capital Management. (arXiv:2205.08435v3 [q-fin.RM] UPDATED)

    [http://arxiv.org/abs/2205.08435](http://arxiv.org/abs/2205.08435)

    本文提出了一个两柱网络风险管理框架，其中包括网络风险评估和网络资本管理。通过基于历史网络事件数据集的案例研究，展示了全面的成本效益分析对于网络风险管理的重要性，并说明最佳策略取决于不同因素。

    

    网络风险是一个在日益数字化的世界中无处不在的风险，众所周知，这种风险很难管理。本文提出了一个两柱网络风险管理框架来解决这个问题。第一柱，网络风险评估，将保险中的频率-严重性模型与网络安全中的级联模型相结合，以捕捉网络风险的独特特征。第二柱，网络资本管理，提供了有关平衡网络风险管理策略的信息决策，包括网络安全投资、保险覆盖和储备金。这个框架通过基于历史网络事件数据集的案例研究进行了演示，表明对于预算有限、目标多样化的公司来说，全面的成本效益分析对于网络风险管理至关重要。敏感性分析还说明了最佳策略取决于诸多因素，如网络安全投资的数量和效果。

    Cyber risk is an omnipresent risk in the increasingly digitized world that is known to be difficult to manage. This paper proposes a two-pillar cyber risk management framework to address such difficulty. The first pillar, cyber risk assessment, blends the frequency-severity model in insurance with the cascade model in cybersecurity, to capture the unique feature of cyber risk. The second pillar, cyber capital management, provides informative decision-making on a balanced cyber risk management strategy, which includes cybersecurity investments, insurance coverage, and reserves. This framework is demonstrated by a case study based on a historical cyber incident dataset, which shows that a comprehensive cost-benefit analysis is necessary for a budget-constrained company with competing objectives for cyber risk management. Sensitivity analysis also illustrates that the best strategy depends on various factors, such as the amount of cybersecurity investments and the effectiveness of cyberse
    

