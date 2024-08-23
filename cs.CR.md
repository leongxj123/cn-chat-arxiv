# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030) | 本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。 |
| [^2] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^3] | [Improving the Utility of Differentially Private Clustering through Dynamical Processing.](http://arxiv.org/abs/2304.13886) | 本研究提出了一种通过利用 Morse 理论提高差分私有聚类效用的方法，该方法可为复杂集群分布适配高斯子集群，即使对于现有的简单聚类方法，其效果也更好，在相同的隐私水平下不会增加隐私损失。 |

# 详细

[^1]: 利用潜在对抗训练防御未预见的故障模式

    Defending Against Unforeseen Failure Modes with Latent Adversarial Training

    [https://arxiv.org/abs/2403.05030](https://arxiv.org/abs/2403.05030)

    本研究利用潜在对抗训练（LAT）来防御AI系统中未预见的故障模式，通过利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示，有效清除了恶意软件和对抗性攻击。

    

    人工智能系统有时在部署后会展示出有害的意外行为。尽管开发人员进行了大量诊断和调试，这种情况经常发生。由于攻击面非常广泛，从模型中减少风险具有挑战性。耗尽地搜索可能导致模型失败的输入是不可行的。红队和对抗训练（AT）通常用于使人工智能系统更加健壮。然而，它们并不足以避免许多与对抗训练不同的真实世界故障模式。在这项工作中，我们利用潜在对抗训练（LAT）来防御漏洞，而无需生成引发这些漏洞的输入。LAT利用网络实际用于预测的压缩、抽象和结构化概念的潜在表示。我们使用LAT来清除恶意软件并防御针对保留类别的对抗性攻击。我们展示在图像分类、文本分类

    arXiv:2403.05030v1 Announce Type: cross  Abstract: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classifi
    
[^2]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^3]: 通过动态处理提高差分私有聚类的效用

    Improving the Utility of Differentially Private Clustering through Dynamical Processing. (arXiv:2304.13886v1 [cs.LG])

    [http://arxiv.org/abs/2304.13886](http://arxiv.org/abs/2304.13886)

    本研究提出了一种通过利用 Morse 理论提高差分私有聚类效用的方法，该方法可为复杂集群分布适配高斯子集群，即使对于现有的简单聚类方法，其效果也更好，在相同的隐私水平下不会增加隐私损失。

    

    本研究旨在缓解差分私有聚类任务中效用和隐私之间的权衡问题。现有研究主要关注简单的集群方法，对于非凸集群的聚类效果较差。通过利用 Morse 理论，我们将高斯子集群按层次连接以适应更复杂的集群分布。由于差分私有子群是通过现有方法获得的，因此所提出的方法几乎不会增加隐私损失。我们提供了理论背景，表明所提出的方法是归纳的，并且可以实现任意数量的群集。在各种数据集上的实验证明，与现有方法相比，在相同的隐私水平下，我们的框架实现了更好的聚类效果。

    This study aims to alleviate the trade-off between utility and privacy in the task of differentially private clustering. Existing works focus on simple clustering methods, which show poor clustering performance for non-convex clusters. By utilizing Morse theory, we hierarchically connect the Gaussian sub-clusters to fit complex cluster distributions. Because differentially private sub-clusters are obtained through the existing methods, the proposed method causes little or no additional privacy loss. We provide a theoretical background that implies that the proposed method is inductive and can achieve any desired number of clusters. Experiments on various datasets show that our framework achieves better clustering performance at the same privacy level, compared to the existing methods.
    

