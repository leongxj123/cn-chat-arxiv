# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Online Federated Learning with Correlated Noise](https://arxiv.org/abs/2403.16542) | 提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。 |
| [^2] | [Reinforcement Unlearning.](http://arxiv.org/abs/2312.15910) | 强化学习中的消除学习是一种新兴的研究领域，旨在解决环境所有者有权撤销智能体训练数据的隐私问题。该领域面临三个主要挑战。 |
| [^3] | [Launching a Robust Backdoor Attack under Capability Constrained Scenarios.](http://arxiv.org/abs/2304.10985) | 深度神经网络的后门攻击一直是一个安全性问题，现有的改进方法需要强大的攻击者能力，在能力受限场景下还没有找到令人满意的解决办法，此外，模型鲁棒性仍然值得关注。 |

# 详细

[^1]: 具有相关噪声的差分隐私在线联邦学习

    Differentially Private Online Federated Learning with Correlated Noise

    [https://arxiv.org/abs/2403.16542](https://arxiv.org/abs/2403.16542)

    提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。

    

    我们提出了一种新颖的差分隐私算法，用于在线联邦学习，利用时间相关的噪声来提高效用同时确保连续发布的模型的隐私性。为了解决源自DP噪声和本地更新带来的流式非独立同分布数据的挑战，我们开发了扰动迭代分析来控制DP噪声对效用的影响。此外，我们展示了在准强凸条件下如何有效管理来自本地更新的漂移误差。在$(\epsilon, \delta)$-DP预算范围内，我们建立了整个时间段上的动态遗憾界，量化了关键参数的影响以及动态环境变化的强度。数值实验证实了所提算法的有效性。

    arXiv:2403.16542v1 Announce Type: new  Abstract: We propose a novel differentially private algorithm for online federated learning that employs temporally correlated noise to improve the utility while ensuring the privacy of the continuously released models. To address challenges stemming from DP noise and local updates with streaming noniid data, we develop a perturbed iterate analysis to control the impact of the DP noise on the utility. Moreover, we demonstrate how the drift errors from local updates can be effectively managed under a quasi-strong convexity condition. Subject to an $(\epsilon, \delta)$-DP budget, we establish a dynamic regret bound over the entire time horizon that quantifies the impact of key parameters and the intensity of changes in dynamic environments. Numerical experiments validate the efficacy of the proposed algorithm.
    
[^2]: 强化学习中的消除学习

    Reinforcement Unlearning. (arXiv:2312.15910v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.15910](http://arxiv.org/abs/2312.15910)

    强化学习中的消除学习是一种新兴的研究领域，旨在解决环境所有者有权撤销智能体训练数据的隐私问题。该领域面临三个主要挑战。

    

    机器消除学习指的是根据数据所有者的请求，降低特定训练数据对机器学习模型的影响的过程。然而，在消除学习的研究中，一个重要的领域往往被忽视，那就是强化学习。强化学习旨在训练一个智能体在环境中做出最优决策以最大化累积奖励。在训练过程中，智能体往往会记忆环境的特征，这引发了一个重大的隐私问题。根据数据保护法规，环境的所有者有权撤销智能体的训练数据的访问权限，因此需要开展一个新颖且紧迫的研究领域，即“强化消除学习”。强化消除学习侧重于撤销整个环境而不是单个数据样本。这一独特特征带来了三个不同的挑战：1）如何提出消除学习方案

    Machine unlearning refers to the process of mitigating the influence of specific training data on machine learning models based on removal requests from data owners. However, one important area that has been largely overlooked in the research of unlearning is reinforcement learning. Reinforcement learning focuses on training an agent to make optimal decisions within an environment to maximize its cumulative rewards. During the training, the agent tends to memorize the features of the environment, which raises a significant concern about privacy. As per data protection regulations, the owner of the environment holds the right to revoke access to the agent's training data, thus necessitating the development of a novel and pressing research field, known as \emph{reinforcement unlearning}. Reinforcement unlearning focuses on revoking entire environments rather than individual data samples. This unique characteristic presents three distinct challenges: 1) how to propose unlearning schemes f
    
[^3]: 在能力受限场景下启动强韧后门攻击

    Launching a Robust Backdoor Attack under Capability Constrained Scenarios. (arXiv:2304.10985v1 [cs.CR])

    [http://arxiv.org/abs/2304.10985](http://arxiv.org/abs/2304.10985)

    深度神经网络的后门攻击一直是一个安全性问题，现有的改进方法需要强大的攻击者能力，在能力受限场景下还没有找到令人满意的解决办法，此外，模型鲁棒性仍然值得关注。

    

    随着深度神经网络在关键领域的应用不断增加，人们开始担心它们的安全性。由于缺乏透明度，深度学习模型容易受到后门攻击的威胁。污染的后门模型在普通环境下可能表现正常，但当输入包含触发器时，会显示出恶意行为。目前对后门攻击的研究集中于改善触发器的秘密性，大多数方法需要强大的攻击者能力，例如对模型结构的了解或对训练过程的控制。由于在大多数情况下攻击者的能力受到限制，这些攻击是不切实际的。此外，模型鲁棒性的问题还未得到充分关注。例如，模型蒸馏常用于简化模型大小，但随着参数数量指数级增长，以前的许多后门攻击在模型蒸馏后均失败;图像增强操作可以破坏触发器，从而使后门攻击失效。

    As deep neural networks continue to be used in critical domains, concerns over their security have emerged. Deep learning models are vulnerable to backdoor attacks due to the lack of transparency. A poisoned backdoor model may perform normally in routine environments, but exhibit malicious behavior when the input contains a trigger. Current research on backdoor attacks focuses on improving the stealthiness of triggers, and most approaches require strong attacker capabilities, such as knowledge of the model structure or control over the training process. These attacks are impractical since in most cases the attacker's capabilities are limited. Additionally, the issue of model robustness has not received adequate attention. For instance, model distillation is commonly used to streamline model size as the number of parameters grows exponentially, and most of previous backdoor attacks failed after model distillation; the image augmentation operations can destroy the trigger and thus disabl
    

