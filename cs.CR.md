# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning.](http://arxiv.org/abs/2310.04055) | 本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。 |
| [^2] | [Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework.](http://arxiv.org/abs/2306.07992) | 本文提出了一种对抗图像重构及检测框架来保护视觉感知推荐系统，能够防御局部扰动为特征的对抗攻击并且能够在干净和对抗性图像上进行训练来检测对抗性图像。 |
| [^3] | [Differentially Private Synthetic Data via Foundation Model APIs 1: Images.](http://arxiv.org/abs/2305.15560) | 该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。 |

# 详细

[^1]: 把坏人踢出去！基于零知识证明的联邦学习异常检测

    Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning. (arXiv:2310.04055v1 [cs.CR])

    [http://arxiv.org/abs/2310.04055](http://arxiv.org/abs/2310.04055)

    本文提出了一种基于零知识证明的联邦学习异常检测方法，实现了在实际系统中检测和消除恶意客户端模型的能力。

    

    联邦学习系统容易受到恶意客户端的攻击，他们通过提交篡改的本地模型来达到对抗目标，比如阻止全局模型的收敛或者导致全局模型对某些数据进行错误分类。许多现有的防御机制在实际联邦学习系统中不可行，因为它们需要先知道恶意客户端的数量，或者依赖重新加权或修改提交的方式。这是因为攻击者通常不会在攻击之前宣布他们的意图，而重新加权可能会改变聚合结果，即使没有攻击。为了解决这些在实际联邦学习系统中的挑战，本文引入了一种最尖端的异常检测方法，具有以下特点：i）仅在发生攻击时检测攻击的发生并进行防御操作；ii）一旦发生攻击，进一步检测恶意客户端模型并将其消除，而不会对正常模型造成伤害；iii）确保

    Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuri
    
[^2]: 安全的视觉感知推荐系统：一种对抗图像重构及检测框架

    Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework. (arXiv:2306.07992v1 [cs.CV])

    [http://arxiv.org/abs/2306.07992](http://arxiv.org/abs/2306.07992)

    本文提出了一种对抗图像重构及检测框架来保护视觉感知推荐系统，能够防御局部扰动为特征的对抗攻击并且能够在干净和对抗性图像上进行训练来检测对抗性图像。

    

    随着富含图片等视觉数据与物品关联度增加，视觉感知推荐系统（VARS）已被广泛应用于不同应用领域。最近的研究表明，VARS易受到物品-图像对抗攻击的攻击，这些攻击向与这些物品关联的干净图像添加人类无法感知的扰动。对VARS的攻击为广泛使用VARS的许多应用（如电子商务和社交网络）带来新的安全挑战。如何保护VARS免受此类对抗攻击成为一个关键的问题。目前，尚缺乏系统地研究如何设计针对VARS视觉攻击的安全防御策略。本文提出了一种对抗图像重构及检测框架来保护VARS，我们的方法可以同时(1)通过基于全局视觉传输的图像重构来防御以局部扰动为特征的对抗攻击，(2)使用在少量干净和对抗性图像上训练的检测模型来检测对抗性图像。实验结果表明，我们的框架能够有效地防御各种物品-图像对抗攻击对VARS的影响。

    With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision tra
    
[^3]: 基于 Foundation Model APIs 的差分隐私合成数据：图片

    Differentially Private Synthetic Data via Foundation Model APIs 1: Images. (arXiv:2305.15560v1 [cs.CV])

    [http://arxiv.org/abs/2305.15560](http://arxiv.org/abs/2305.15560)

    该论文提出了基于API的方法生成密切类似于原始私有数据的差分隐私（DP）合成数据，可以更轻松地部署。使用Private Evolution（PE）框架生成DP合成图像，结合了差分隐私、进化算法和元学习的技术，可以在保护隐私的同时生成既为DP又与原始图像外观相似的合成图像，并在流行的图像数据集上表现优异。

    

    在当前数据驱动的世界中，生成密切类似于原始私有数据的差分隐私（DP）合成数据是一种可扩展的方法，可减轻隐私问题。与当前为此任务训练定制模型的做法相反，我们旨在通过API生成DP合成数据（DPSDA），其中我们将基础模型视为黑盒并只利用其推理API。这些基于API的、无需训练的方法更容易部署，如最近 API 应用程序的激增所证明的那样。这些方法还可以利用可通过其推理API访问其权重未发布的大型基础模型的能力。但是，由于模型访问更加严格，还需保护API提供商的隐私，这将带来更大的挑战。在本文中，我们提出了一个称为 Private Evolution（PE）的新框架，以解决这个问题，并展示了其在使用基础模型API生成DP合成图像方面的初始实现。PE结合了差分隐私、进化算法和元学习的技术，有效地生成既为DP又与原始图像外观相似的合成图像。我们还在流行的图像数据集如CIFAR-10上评估了我们的框架，并显示我们的方法在效用和隐私方面优于现有的DP图像生成方法。

    Generating differentially private (DP) synthetic data that closely resembles the original private data without leaking sensitive user information is a scalable way to mitigate privacy concerns in the current data-driven world. In contrast to current practices that train customized models for this task, we aim to generate DP Synthetic Data via APIs (DPSDA), where we treat foundation models as blackboxes and only utilize their inference APIs. Such API-based, training-free approaches are easier to deploy as exemplified by the recent surge in the number of API-based apps. These approaches can also leverage the power of large foundation models which are accessible via their inference APIs while the model weights are unreleased. However, this comes with greater challenges due to strictly more restrictive model access and the additional need to protect privacy from the API provider.  In this paper, we present a new framework called Private Evolution (PE) to solve this problem and show its ini
    

