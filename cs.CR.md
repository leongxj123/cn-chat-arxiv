# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships](https://arxiv.org/abs/2402.12189) | 攻击者通过对预训练LM进行对抗微调，以放大原始训练数据的曝光，采用伪标签和机器生成概率来加强LM对预训练数据的保留。 |
| [^2] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^3] | [MIRA: Cracking Black-box Watermarking on Deep Neural Networks via Model Inversion-based Removal Attacks.](http://arxiv.org/abs/2309.03466) | 本文提出了一种基于模型反演的去除攻击方法（\textsc{Mira}），该方法能够有效地破解大多数主流黑盒DNN水印方案，通过利用受保护模型的内部信息来恢复和消除水印信息。 |

# 详细

[^1]: 通过使用伪标签成员资格进行微调来增强训练数据曝光

    Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships

    [https://arxiv.org/abs/2402.12189](https://arxiv.org/abs/2402.12189)

    攻击者通过对预训练LM进行对抗微调，以放大原始训练数据的曝光，采用伪标签和机器生成概率来加强LM对预训练数据的保留。

    

    神经语言模型(LMs)由于数据记忆而容易受到训练数据提取攻击的影响。本文介绍了一种新的攻击场景，在这种场景中，攻击者对预训练LM进行对抗微调，以放大原始训练数据的曝光。该策略不同于先前的研究，其目的是加强LM对其预训练数据集的保留。为了实现这一目标，攻击者需要收集与预训练数据密切相关的生成文本。然而，如果没有实际数据集的知识，衡量生成文本中预训练数据的量是具有挑战性的。为了解决这个问题，我们提出利用目标LM的机器生成概率所表示的成员近似值为这些生成文本使用伪标签。随后，我们微调LM以支持那些更有可能源自预训练数据的生成文本，根据其成员资格。

    arXiv:2402.12189v1 Announce Type: new  Abstract: Neural language models (LMs) are vulnerable to training data extraction attacks due to data memorization. This paper introduces a novel attack scenario wherein an attacker adversarially fine-tunes pre-trained LMs to amplify the exposure of the original training data. This strategy differs from prior studies by aiming to intensify the LM's retention of its pre-training dataset. To achieve this, the attacker needs to collect generated texts that are closely aligned with the pre-training data. However, without knowledge of the actual dataset, quantifying the amount of pre-training data within generated texts is challenging. To address this, we propose the use of pseudo-labels for these generated texts, leveraging membership approximations indicated by machine-generated probabilities from the target LM. We subsequently fine-tune the LM to favor generations with higher likelihoods of originating from the pre-training data, based on their memb
    
[^2]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^3]: 利用基于模型反演的去除攻击方法破解深度神经网络中的黑盒水印

    MIRA: Cracking Black-box Watermarking on Deep Neural Networks via Model Inversion-based Removal Attacks. (arXiv:2309.03466v1 [cs.CR])

    [http://arxiv.org/abs/2309.03466](http://arxiv.org/abs/2309.03466)

    本文提出了一种基于模型反演的去除攻击方法（\textsc{Mira}），该方法能够有效地破解大多数主流黑盒DNN水印方案，通过利用受保护模型的内部信息来恢复和消除水印信息。

    

    为了保护训练有素的深度神经网络（DNN）的知识产权，黑盒DNN水印已在学术界和工业界越来越受欢迎。这些水印被嵌入到DNN模型在一组特别设计的样本上的预测行为中。最近的研究经验证明，大多数黑盒水印方案对已知的去除攻击具有抵抗能力。本文提出了一种新颖的基于模型反演的去除攻击方法（\textsc{Mira}），该方法对大多数主流黑盒DNN水印方案都是无关水印的，并且具有高效性。总的来说，我们的攻击流程利用受保护模型的内部信息来恢复和消除水印信息。我们还设计了目标类别检测和恢复样本分割算法，以减少\textsc{Mira}引起的效用损失，并实现最优的攻击效果。

    To protect the intellectual property of well-trained deep neural networks (DNNs), black-box DNN watermarks, which are embedded into the prediction behavior of DNN models on a set of specially-crafted samples, have gained increasing popularity in both academy and industry. Watermark robustness is usually implemented against attackers who steal the protected model and obfuscate its parameters for watermark removal. Recent studies empirically prove the robustness of most black-box watermarking schemes against known removal attempts.  In this paper, we propose a novel Model Inversion-based Removal Attack (\textsc{Mira}), which is watermark-agnostic and effective against most of mainstream black-box DNN watermarking schemes. In general, our attack pipeline exploits the internals of the protected model to recover and unlearn the watermark message. We further design target class detection and recovered sample splitting algorithms to reduce the utility loss caused by \textsc{Mira} and achieve 
    

