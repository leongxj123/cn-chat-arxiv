# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provably Robust Cost-Sensitive Learning via Randomized Smoothing.](http://arxiv.org/abs/2310.08732) | 本研究通过随机平滑认证框架，为成本敏感的稳健分类器提供了严格的稳健性保证，并通过优化方案针对不同数据子组设计了细粒度认证半径，取得了优越的性能。 |
| [^2] | [Refiner: Data Refining against Gradient Leakage Attacks in Federated Learning.](http://arxiv.org/abs/2212.02042) | Refiner提出了一种创新的防御范式，通过构建与原始数据具有低语义相似性的健壮数据，有效地混淆梯度泄漏攻击者，从而提高联邦学习系统的隐私保护能力。 |

# 详细

[^1]: 可证保健康商务字体尹包通过随机平滑(译注)水。

    Provably Robust Cost-Sensitive Learning via Randomized Smoothing. (arXiv:2310.08732v1 [cs.LG])

    [http://arxiv.org/abs/2310.08732](http://arxiv.org/abs/2310.08732)

    本研究通过随机平滑认证框架，为成本敏感的稳健分类器提供了严格的稳健性保证，并通过优化方案针对不同数据子组设计了细粒度认证半径，取得了优越的性能。

    

    我们关注于在成本敏感的情景下学习对抗性稳健分类器，在这种情况下，不同类别的对抗性变换的潜在危害被编码在一个二进制成本矩阵中。现有的方法要么是经验性的，无法证明稳健性，要么存在固有的可扩展性问题。在这项工作中，我们研究了随机平滑，一种更可扩展的稳健性认证框架，是否可以用于证明成本敏感的稳健性。建立在一种成本敏感认证半径的概念之上，我们展示了如何调整标准的随机平滑认证流程，为任何成本矩阵产生严格的稳健性保证。此外，通过针对不同数据子组设计的细粒度认证半径优化方案，我们提出了一种算法，用于训练针对成本敏感稳健性优化的平滑分类器。在图像基准测试和真实的医学数据集上进行了大量实验，证明了我们方法的优越性。

    We focus on learning adversarially robust classifiers under a cost-sensitive scenario, where the potential harm of different classwise adversarial transformations is encoded in a binary cost matrix. Existing methods are either empirical that cannot certify robustness or suffer from inherent scalability issues. In this work, we study whether randomized smoothing, a more scalable robustness certification framework, can be leveraged to certify cost-sensitive robustness. Built upon a notion of cost-sensitive certified radius, we show how to adapt the standard randomized smoothing certification pipeline to produce tight robustness guarantees for any cost matrix. In addition, with fine-grained certified radius optimization schemes specifically designed for different data subgroups, we propose an algorithm to train smoothed classifiers that are optimized for cost-sensitive robustness. Extensive experiments on image benchmarks and a real-world medical dataset demonstrate the superiority of our
    
[^2]: Refiner: 针对联邦学习中的梯度泄漏攻击的数据精炼方法

    Refiner: Data Refining against Gradient Leakage Attacks in Federated Learning. (arXiv:2212.02042v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02042](http://arxiv.org/abs/2212.02042)

    Refiner提出了一种创新的防御范式，通过构建与原始数据具有低语义相似性的健壮数据，有效地混淆梯度泄漏攻击者，从而提高联邦学习系统的隐私保护能力。

    

    最近的研究引起了对联邦学习系统易受梯度泄漏攻击的关注。这类攻击利用客户端上传的梯度来重构其敏感数据，从而破坏了联邦学习的隐私保护能力。为了应对这一威胁，已经提出了各种防御机制来减轻攻击的影响，这些机制通过操纵上传的梯度来防止攻击。然而，实证评估表明这些防御措施在面对复杂攻击时具有有限的弹性，这表明迫切需要更有效的防御方法。本文提出了一种新的防御范式，不同于传统的梯度扰动方法，而是专注于构建健壮数据。直观地说，如果健壮数据与客户端原始数据具有很低的语义相似性，与健壮数据相关的梯度可以有效地混淆攻击者。为此，我们设计了Refiner，它同时优化了两个指标，用于隐私保护和...

    Recent works have brought attention to the vulnerability of Federated Learning (FL) systems to gradient leakage attacks. Such attacks exploit clients' uploaded gradients to reconstruct their sensitive data, thereby compromising the privacy protection capability of FL. In response, various defense mechanisms have been proposed to mitigate this threat by manipulating the uploaded gradients. Unfortunately, empirical evaluations have demonstrated limited resilience of these defenses against sophisticated attacks, indicating an urgent need for more effective defenses. In this paper, we explore a novel defensive paradigm that departs from conventional gradient perturbation approaches and instead focuses on the construction of robust data. Intuitively, if robust data exhibits low semantic similarity with clients' raw data, the gradients associated with robust data can effectively obfuscate attackers. To this end, we design Refiner that jointly optimizes two metrics for privacy protection and 
    

