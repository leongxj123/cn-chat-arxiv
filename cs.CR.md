# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Visual Privacy Auditing with Diffusion Models](https://arxiv.org/abs/2403.07588) | 在这项研究中，通过使用扩散模型进行重建攻击，作者发现在DP-SGD下，真实世界的数据先验对于重建成功具有显著影响。 |
| [^2] | [Continual Adversarial Defense](https://arxiv.org/abs/2312.09481) | 提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。 |

# 详细

[^1]: 基于扩散模型的视觉隐私审计

    Visual Privacy Auditing with Diffusion Models

    [https://arxiv.org/abs/2403.07588](https://arxiv.org/abs/2403.07588)

    在这项研究中，通过使用扩散模型进行重建攻击，作者发现在DP-SGD下，真实世界的数据先验对于重建成功具有显著影响。

    

    arXiv:2403.07588v1 声明类型: 新的 摘要: 对机器学习模型的图像重建攻击可能会导致泄露敏感信息，从而对隐私构成重大风险。虽然使用差分隐私(DP)来抵御此类攻击已被证明是有效的，但确定适当的DP参数仍然具有挑战性。当前对数据重建成功的形式化保证受到了关于对手对目标数据的了解的过于理论化的假设的影响，特别是在图像领域。在这项工作中，我们通过实证调查这一差异，并发现这些假设的实际性在很大程度上取决于数据先验和重建目标之间的域转移。我们提出了一种基于扩散模型(DMs)的重建攻击，假设对手可以访问真实世界的图像先验，并评估其对在DP-SGD下的隐私泄露的影响。我们展示了(1)真实世界的数据先验显著影响重建成功，

    arXiv:2403.07588v1 Announce Type: new  Abstract: Image reconstruction attacks on machine learning models pose a significant risk to privacy by potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) has proven effective, determining appropriate DP parameters remains challenging. Current formal guarantees on data reconstruction success suffer from overly theoretical assumptions regarding adversary knowledge about the target data, particularly in the image domain. In this work, we empirically investigate this discrepancy and find that the practicality of these assumptions strongly depends on the domain shift between the data prior and the reconstruction target. We propose a reconstruction attack based on diffusion models (DMs) that assumes adversary access to real-world image priors and assess its implications on privacy leakage under DP-SGD. We show that (1) real-world data priors significantly influence reconstruction success, 
    
[^2]: 持续不断的对抗性防御

    Continual Adversarial Defense

    [https://arxiv.org/abs/2312.09481](https://arxiv.org/abs/2312.09481)

    提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架。

    

    针对每月针对视觉分类器的对抗性攻击快速演变的特性，人们提出了许多防御方法，旨在尽可能通用化以抵御尽可能多的已知攻击。然而，设计一个能够对抗所有类型攻击的防御方法并不现实，因为防御系统运行的环境是动态的，包含随着时间出现的各种独特攻击。防御系统必须收集在线少样本对抗反馈以迅速增强自身，充分利用内存。因此，我们提出了第一个能够动态适应任何攻击的持续对抗性防御（CAD）框架，其中各种攻击逐个阶段出现。在实践中，CAD基于四项原则进行建模：(1) 持续适应新攻击而无灾难性遗忘，(2) 少样本适应，(3) 内存高效适应，以及(4) 高准确性

    arXiv:2312.09481v2 Announce Type: replace-cross  Abstract: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accur
    

