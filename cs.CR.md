# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains](https://arxiv.org/abs/2403.06672) | 本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。 |
| [^2] | [Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes](https://arxiv.org/abs/2403.00867) | 本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。 |
| [^3] | [Differentially Private Range Queries with Correlated Input Perturbation](https://arxiv.org/abs/2402.07066) | 本研究提出了一种具有相关输入扰动的差分隐私范围查询的局部机制，通过级联采样算法实现，实验表明在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。 |

# 详细

[^1]: 在隐私敏感领域中从联邦学习中有可证明的互惠益处

    Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains

    [https://arxiv.org/abs/2403.06672](https://arxiv.org/abs/2403.06672)

    本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。

    

    跨领域联邦学习（FL）允许数据所有者通过从彼此的私有数据集中获益来训练准确的机器学习模型。本文研究了在何时以及如何服务器可以设计一种对所有参与者都有利的FL协议的问题。我们提供了在均值估计和凸随机优化背景下存在相互有利协议的必要和充分条件。我们推导出了在对称隐私偏好下，最大化总客户效用的协议。最后，我们设计了最大化最终模型准确性的协议，并在合成实验中展示了它们的好处。

    arXiv:2403.06672v1 Announce Type: cross  Abstract: Cross-silo federated learning (FL) allows data owners to train accurate machine learning models by benefiting from each others private datasets. Unfortunately, the model accuracy benefits of collaboration are often undermined by privacy defenses. Therefore, to incentivize client participation in privacy-sensitive domains, a FL protocol should strike a delicate balance between privacy guarantees and end-model accuracy. In this paper, we study the question of when and how a server could design a FL protocol provably beneficial for all participants. First, we provide necessary and sufficient conditions for the existence of mutually beneficial protocols in the context of mean estimation and convex stochastic optimization. We also derive protocols that maximize the total clients' utility, given symmetric privacy preferences. Finally, we design protocols maximizing end-model accuracy and demonstrate their benefits in synthetic experiments.
    
[^2]: 梯度被罚：通过探索拒绝损失地形图来检测针对大语言模型的越狱攻击

    Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes

    [https://arxiv.org/abs/2403.00867](https://arxiv.org/abs/2403.00867)

    本文提出了一种名为Gradient Cuff的方法，通过探索拒绝损失地形图来检测对大语言模型的越狱攻击，成功设计了一种有效的两步检测策略。

    

    大型语言模型（LLMs）正成为一种突出的生成式AI工具，用户输入查询，LLM生成答案。为了减少伤害和滥用，人们通过使用先进的训练技术如来自人类反馈的强化学习（RLHF）来将这些LLMs与人类价值观保持一致。然而，最近的研究突显了LLMs对于试图颠覆嵌入的安全防护措施的对抗性越狱尝试的脆弱性。为了解决这一挑战，本文定义并调查了LLMs的拒绝损失，然后提出了一种名为Gradient Cuff的方法来检测越狱尝试。Gradient Cuff利用拒绝损失地形图中观察到的独特特性，包括功能值及其光滑性，设计了一种有效的两步检测策略。

    arXiv:2403.00867v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN,
    
[^3]: 具有相关输入扰动的差分隐私范围查询

    Differentially Private Range Queries with Correlated Input Perturbation

    [https://arxiv.org/abs/2402.07066](https://arxiv.org/abs/2402.07066)

    本研究提出了一种具有相关输入扰动的差分隐私范围查询的局部机制，通过级联采样算法实现，实验表明在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。

    

    本工作提出了一种用于线性查询的局部差分隐私机制，特别是范围查询，利用相关输入扰动同时实现无偏性、一致性、统计透明性和对精度目标的控制，无论是在某些查询边缘上还是在层次数据库结构所暗示的精度要求上。所提出的级联采样算法准确高效地实现了该机制。我们的界限表明，我们在保障近乎最优的效用的同时，与输出扰动方法在实践中具有竞争力。

    This work proposes a class of locally differentially private mechanisms for linear queries, in particular range queries, that leverages correlated input perturbation to simultaneously achieve unbiasedness, consistency, statistical transparency, and control over utility requirements in terms of accuracy targets expressed either in certain query margins or as implied by the hierarchical database structure. The proposed Cascade Sampling algorithm instantiates the mechanism exactly and efficiently. Our bounds show that we obtain near-optimal utility while being empirically competitive against output perturbation methods.
    

