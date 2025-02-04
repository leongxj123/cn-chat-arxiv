# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ADVREPAIR:Provable Repair of Adversarial Attack](https://arxiv.org/abs/2404.01642) | ADVREPAIR是一种利用有限数据进行对抗攻击的可证修复的新方法，通过形式验证构建补丁模块，在稳健邻域内提供可证和专门的修复，同时具有泛化到其他输入的防御能力。 |
| [^2] | [Stochastic Gradient Langevin Unlearning](https://arxiv.org/abs/2403.17105) | 本工作提出了随机梯度 Langevin 反遗忘方法，为近似反遗忘问题提供了隐私保障，并展示了小批次梯度更新相较于全批次的优越性能。 |
| [^3] | [Large Language Models are Advanced Anonymizers](https://arxiv.org/abs/2402.13846) | 大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。 |
| [^4] | [Evil from Within: Machine Learning Backdoors through Hardware Trojans.](http://arxiv.org/abs/2304.08411) | 本文介绍了一种在常见机器学习硬件加速器内的后门攻击方法，将最小后门概念和可配置的硬件木马结合使用，从而对目前的防御措施构成挑战。 |

# 详细

[^1]: ADVREPAIR：对抗攻击的可证修复

    ADVREPAIR:Provable Repair of Adversarial Attack

    [https://arxiv.org/abs/2404.01642](https://arxiv.org/abs/2404.01642)

    ADVREPAIR是一种利用有限数据进行对抗攻击的可证修复的新方法，通过形式验证构建补丁模块，在稳健邻域内提供可证和专门的修复，同时具有泛化到其他输入的防御能力。

    

    深度神经网络(DNNs)在安全关键领域中的部署日益增加，但它们对抗性攻击的脆弱性构成严重的安全风险。现有的使用有限数据的神经元级方法在修复对手方面缺乏效力，因为对抗攻击机制的固有复杂性，而对抗训练，利用大量对抗样本增强鲁棒性，缺乏可证性。在本文中，我们提出了ADVREPAIR，一种利用有限数据进行对抗攻击的可证修复的新方法。通过利用形式验证，ADVREPAIR构建补丁模块，当与原始网络集成时，在稳健邻域内提供可证和专门的修复。此外，我们的方法还包括一种启发式机制来分配补丁模块，使得这种防御对抗攻击泛化到其他输入。ADVREPAIR展示了卓越的效率。

    arXiv:2404.01642v1 Announce Type: new  Abstract: Deep neural networks (DNNs) are increasingly deployed in safety-critical domains, but their vulnerability to adversarial attacks poses serious safety risks. Existing neuron-level methods using limited data lack efficacy in fixing adversaries due to the inherent complexity of adversarial attack mechanisms, while adversarial training, leveraging a large number of adversarial samples to enhance robustness, lacks provability. In this paper, we propose ADVREPAIR, a novel approach for provable repair of adversarial attacks using limited data. By utilizing formal verification, ADVREPAIR constructs patch modules that, when integrated with the original network, deliver provable and specialized repairs within the robustness neighborhood. Additionally, our approach incorporates a heuristic mechanism for assigning patch modules, allowing this defense against adversarial attacks to generalize to other inputs. ADVREPAIR demonstrates superior efficienc
    
[^2]: 随机梯度 Langevin 反遗忘

    Stochastic Gradient Langevin Unlearning

    [https://arxiv.org/abs/2403.17105](https://arxiv.org/abs/2403.17105)

    本工作提出了随机梯度 Langevin 反遗忘方法，为近似反遗忘问题提供了隐私保障，并展示了小批次梯度更新相较于全批次的优越性能。

    

    “被遗忘的权利”是用户数据隐私的法律所确保的越来越重要。机器反遗忘旨在高效地消除已训练模型参数上某些数据点的影响，使其近似于从头开始重新训练模型。本研究提出了随机梯度 Langevin 反遗忘，这是第一个基于带有隐私保障的噪声随机梯度下降（SGD）的反遗忘框架，适用于凸性假设下的近似反遗忘问题。我们的结果表明，与全批次对应方法相比，小批次梯度更新在隐私复杂度权衡方面提供了更好的性能。我们的反遗忘方法具有诸多算法优势，包括与重新训练相比的复杂度节省，以及支持顺序和批量反遗忘。为了检验我们方法的隐私-效用-复杂度权衡，我们在基准数据集上进行了实验比较。

    arXiv:2403.17105v1 Announce Type: new  Abstract: ``The right to be forgotten'' ensured by laws for user data privacy becomes increasingly important. Machine unlearning aims to efficiently remove the effect of certain data points on the trained model parameters so that it can be approximately the same as if one retrains the model from scratch. This work proposes stochastic gradient Langevin unlearning, the first unlearning framework based on noisy stochastic gradient descent (SGD) with privacy guarantees for approximate unlearning problems under convexity assumption. Our results show that mini-batch gradient updates provide a superior privacy-complexity trade-off compared to the full-batch counterpart. There are numerous algorithmic benefits of our unlearning approach, including complexity saving compared to retraining, and supporting sequential and batch unlearning. To examine the privacy-utility-complexity trade-off of our method, we conduct experiments on benchmark datasets compared 
    
[^3]: 大型语言模型是先进的匿名化工具

    Large Language Models are Advanced Anonymizers

    [https://arxiv.org/abs/2402.13846](https://arxiv.org/abs/2402.13846)

    大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。

    

    最近在隐私研究领域对大型语言模型的研究表明，它们在推断真实世界在线文本中的个人数据方面表现出接近人类水平的性能。随着模型能力的不断增强，现有的文本匿名化方法当前已经落后于监管要求和对抗威胁。这引出了一个问题：个人如何有效地保护他们在分享在线文本时的个人数据。在这项工作中，我们采取了两步来回答这个问题：首先，我们提出了一个新的设置，用于评估面对对抗性LLM的推断时的匿名化效果，从而允许自然地测量匿名化性能，同时纠正了以前指标的一些缺陷。然后，我们提出了基于LLM的对抗性匿名化框架，利用LLM的强大推断能力来指导我们的匿名化过程。在我们的实验评估中，我们展示了在真实世界中的匿名化实践。

    arXiv:2402.13846v1 Announce Type: cross  Abstract: Recent work in privacy research on large language models has shown that they achieve near human-level performance at inferring personal data from real-world online texts. With consistently increasing model capabilities, existing text anonymization methods are currently lacking behind regulatory requirements and adversarial threats. This raises the question of how individuals can effectively protect their personal data in sharing online texts. In this work, we take two steps to answer this question: We first present a new setting for evaluating anonymizations in the face of adversarial LLMs inferences, allowing for a natural measurement of anonymization performance while remedying some of the shortcomings of previous metrics. We then present our LLM-based adversarial anonymization framework leveraging the strong inferential capabilities of LLMs to inform our anonymization procedure. In our experimental evaluation, we show on real-world 
    
[^4]: 来自内部的邪恶: 通过硬件木马进行机器学习后门攻击

    Evil from Within: Machine Learning Backdoors through Hardware Trojans. (arXiv:2304.08411v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2304.08411](http://arxiv.org/abs/2304.08411)

    本文介绍了一种在常见机器学习硬件加速器内的后门攻击方法，将最小后门概念和可配置的硬件木马结合使用，从而对目前的防御措施构成挑战。

    

    后门会对机器学习造成严重威胁，因为它们可能破坏安全关键的系统，如自动驾驶汽车。本文介绍了一种后门攻击方法，完全居于用于机器学习的常见硬件加速器内，从而对当前防御措施构成挑战。为了使这种攻击实用，我们克服了两个挑战：首先，由于硬件加速器上的存储空间严重受限，因此我们引入了所谓的最小后门概念，只改变少量模型参数即可激活后门。其次，我们开发了一种可配置的硬件木马，可以与后门一起使用。

    Backdoors pose a serious threat to machine learning, as they can compromise the integrity of security-critical systems, such as self-driving cars. While different defenses have been proposed to address this threat, they all rely on the assumption that the hardware on which the learning models are executed during inference is trusted. In this paper, we challenge this assumption and introduce a backdoor attack that completely resides within a common hardware accelerator for machine learning. Outside of the accelerator, neither the learning model nor the software is manipulated, so that current defenses fail. To make this attack practical, we overcome two challenges: First, as memory on a hardware accelerator is severely limited, we introduce the concept of a minimal backdoor that deviates as little as possible from the original model and is activated by replacing a few model parameters only. Second, we develop a configurable hardware trojan that can be provisioned with the backdoor and p
    

