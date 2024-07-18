# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks](https://arxiv.org/abs/2312.14440) | 本文对文本到图像生成中的对抗攻击进行了实证研究，发现了攻击成功率的不对称性，并提出了用于探测模型对抗攻击信号的指标。 |
| [^2] | [DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks.](http://arxiv.org/abs/2306.09124) | DIFFender是一种基于扩散的对抗性防御方法，通过定位和恢复两个阶段的操作，利用文本引导的扩散模型来防御对抗性Patch，从而提高其整体防御性能。 |
| [^3] | [Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension.](http://arxiv.org/abs/2305.15203) | 本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。 |
| [^4] | [SSL-Cleanse: Trojan Detection and Mitigation in Self-Supervised Learning.](http://arxiv.org/abs/2303.09079) | 本篇论文讨论了自监督学习中的木马攻击检测和缓解问题。由于这种攻击危险隐匿，且在下游分类器中很难检测出来。目前在超监督学习中的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。 |

# 详细

[^1]: 对文本到图像生成中的不对称偏差的对抗攻击研究

    Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks

    [https://arxiv.org/abs/2312.14440](https://arxiv.org/abs/2312.14440)

    本文对文本到图像生成中的对抗攻击进行了实证研究，发现了攻击成功率的不对称性，并提出了用于探测模型对抗攻击信号的指标。

    

    文本到图像（T2I）模型在内容生成中的广泛应用需要仔细研究它们的安全性，包括它们对抗攻击的鲁棒性。尽管对抗攻击的研究已经很广泛，但其有效性的原因仍然未被深入探索。本文对T2I模型的对抗攻击进行了实证研究，重点分析了与攻击成功率（ASR）相关的因素。我们引入了一种新的攻击目标 - 使用对抗性后缀进行实体替换，以及两种基于梯度的攻击算法。人工和自动评估揭示了实体交换中ASR的不对称性质：例如，对于在提示“在雨中跳舞的人类”中替换“人类”为“机器人”的对抗性后缀，较容易实现，而反向替换则明显困难得多。我们进一步提出了探测指标，以确定模型对抗ASR的信号。我们发现了：

    arXiv:2312.14440v2 Announce Type: replace Abstract: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We iden
    
[^2]: DIFFender：基于扩散的对抗性防御方法用于抵御Patch攻击

    DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks. (arXiv:2306.09124v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.09124](http://arxiv.org/abs/2306.09124)

    DIFFender是一种基于扩散的对抗性防御方法，通过定位和恢复两个阶段的操作，利用文本引导的扩散模型来防御对抗性Patch，从而提高其整体防御性能。

    

    对抗性攻击，尤其是Patch攻击，对深度学习模型的鲁棒性和可靠性构成了重大威胁。开发可靠的防御方法以抵御Patch攻击对于实际应用至关重要，然而当前在这个领域的研究还不令人满意。在本文中，我们提出了DIFFender，一种新颖的防御方法，它利用文本引导的扩散模型来防御对抗性Patch。DIFFender包括两个主要阶段：Patch定位和Patch恢复。在定位阶段，我们发现并利用了扩散模型的一个有趣特性，以有效地识别对抗性Patch的位置。在恢复阶段，我们利用扩散模型重建图像中的对抗性区域同时保持视觉内容的完整性。重要的是，这两个阶段都受到统一的扩散模型的精心引导，因此我们可以利用它们之间的紧密相互作用来提高整个防御性能。

    Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is not satisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Importantly, these two stages are carefully guided by a unified diffusion model, thus we can utilize the close interaction between them to improve the whole defense performance. Mor
    
[^3]: 通过内在维度将隐性偏见和对抗性攻击相关联

    Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension. (arXiv:2305.15203v1 [cs.LG])

    [http://arxiv.org/abs/2305.15203](http://arxiv.org/abs/2305.15203)

    本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。

    

    尽管神经网络在分类方面表现出色，但众所周知它们易受对抗性攻击的影响。这些攻击是针对模型的输入数据进行的小干扰，旨在欺骗模型。自然而然的问题是，模型的结构、设置或属性与攻击的性质之间可能存在潜在联系。在本文中，我们旨在通过关注神经网络的隐性偏差来解决这个问题，这指的是其固有倾向于支持特定模式或结果。具体而言，我们研究了隐性偏差的一个方面，其中包括进行准确图像分类所需的基本傅里叶频率。我们进行测试以评估这些频率与成功攻击所需的频率之间的统计关系。为了深入探讨这种关系，我们提出了一种新的方法，可以揭示坐标集之间的非线性相关性，在我们的情况下，这些坐标集就是前述的傅里叶频率。

    Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementio
    
[^4]: SSL清理：自监督学习中的木马检测和缓解

    SSL-Cleanse: Trojan Detection and Mitigation in Self-Supervised Learning. (arXiv:2303.09079v1 [cs.CR])

    [http://arxiv.org/abs/2303.09079](http://arxiv.org/abs/2303.09079)

    本篇论文讨论了自监督学习中的木马攻击检测和缓解问题。由于这种攻击危险隐匿，且在下游分类器中很难检测出来。目前在超监督学习中的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。

    

    自监督学习（SSL）是一种常用的学习和编码数据表示的方法。通过使用预先训练的SSL图像编码器并在其顶部训练下游分类器，可以在各种任务上实现令人印象深刻的性能，而只需很少的标记数据。SSL的增加使用导致了与SSL编码器相关的安全研究和各种木马攻击的发展。在SSL编码器中插入木马攻击的危险在于它们能够隐蔽地操作并在各种用户和设备之间广泛传播。Trojaned编码器中的后门行为的存在可能会被下游分类器意外继承，使检测和缓解威胁变得更加困难。虽然超监督学习中当前的木马检测方法可以潜在地保护SSL下游分类器，但在其广泛传播之前识别和处理SSL编码器中的触发器是一项艰巨的任务。

    Self-supervised learning (SSL) is a commonly used approach to learning and encoding data representations. By using a pre-trained SSL image encoder and training a downstream classifier on top of it, impressive performance can be achieved on various tasks with very little labeled data. The increasing usage of SSL has led to an uptick in security research related to SSL encoders and the development of various Trojan attacks. The danger posed by Trojan attacks inserted in SSL encoders lies in their ability to operate covertly and spread widely among various users and devices. The presence of backdoor behavior in Trojaned encoders can inadvertently be inherited by downstream classifiers, making it even more difficult to detect and mitigate the threat. Although current Trojan detection methods in supervised learning can potentially safeguard SSL downstream classifiers, identifying and addressing triggers in the SSL encoder before its widespread dissemination is a challenging task. This is be
    

