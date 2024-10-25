# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reward Poisoning Attack Against Offline Reinforcement Learning](https://arxiv.org/abs/2402.09695) | 这项研究针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题，提出了一种名为“策略对比攻击”的攻击策略。通过使低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的，我们证明了这种攻击有效性。 |
| [^2] | [Inverting Cryptographic Hash Functions via Cube-and-Conquer.](http://arxiv.org/abs/2212.02405) | 该研究应用了Cube-and-Conquer方法将MD4和MD5的步骤缩减版本进行反转。通过逐步修改Dobbertin约束来生成MD4的反转问题，并使用立方阶段的立方和征服方法进行反转。 |

# 详细

[^1]: 对离线强化学习中的奖励污染攻击的研究

    Reward Poisoning Attack Against Offline Reinforcement Learning

    [https://arxiv.org/abs/2402.09695](https://arxiv.org/abs/2402.09695)

    这项研究针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题，提出了一种名为“策略对比攻击”的攻击策略。通过使低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的，我们证明了这种攻击有效性。

    

    我们研究了针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题。我们考虑了一个黑盒威胁模型，攻击者对学习算法完全不了解，并且其预算受到限制，限制了每个数据点的污染量以及总扰动。我们提出了一种名为“策略对比攻击”的攻击策略。其高层思想是使一些低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的。据我们所知，我们首次提出了一种适用于一般离线强化学习场景的黑盒奖励污染攻击。我们提供了关于攻击设计的理论洞察，并在不同类型的学习数据集上经验证明我们的攻击对当前最先进的离线强化学习算法是有效的。

    arXiv:2402.09695v1 Announce Type: cross  Abstract: We study the problem of reward poisoning attacks against general offline reinforcement learning with deep neural networks for function approximation. We consider a black-box threat model where the attacker is completely oblivious to the learning algorithm and its budget is limited by constraining both the amount of corruption at each data point, and the total perturbation. We propose an attack strategy called `policy contrast attack'. The high-level idea is to make some low-performing policies appear as high-performing while making high-performing policies appear as low-performing. To the best of our knowledge, we propose the first black-box reward poisoning attack in the general offline RL setting. We provide theoretical insights on the attack design and empirically show that our attack is efficient against current state-of-the-art offline RL algorithms in different kinds of learning datasets.
    
[^2]: 通过立方和征服法反转密码哈希函数

    Inverting Cryptographic Hash Functions via Cube-and-Conquer. (arXiv:2212.02405v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2212.02405](http://arxiv.org/abs/2212.02405)

    该研究应用了Cube-and-Conquer方法将MD4和MD5的步骤缩减版本进行反转。通过逐步修改Dobbertin约束来生成MD4的反转问题，并使用立方阶段的立方和征服方法进行反转。

    

    MD4和MD5是在1990年代初提出的具有里程碑意义的密码哈希函数。MD4由48个步骤组成，给定任意有限大小的消息，它可以产生一个128位的哈希值。MD5是MD4的更安全的64步扩展。尽管MD4和MD5都容易受到碰撞攻击的影响，但是翻转它们，即通过哈希值找到原始消息仍然不现实。在2007年，MD4的39步版本通过化简为SAT并应用CDCL求解器以及所谓的Dobbertin约束被反转。至于MD5，在2012年，它的28步版本通过CDCL求解器仅针对一个特定的哈希值被反转，而不加任何额外的约束。本研究将立方和征服（CDCL与先行搜索的组合）应用于步骤缩减版本的MD4和MD5的反转。为此，提出了两个算法。第一个算法通过逐步修改Dobbertin约束来生成MD4的反转问题。第二个算法尝试使用立方阶段的立方和征服方法进行反转。

    MD4 and MD5 are seminal cryptographic hash functions proposed in early 1990s. MD4 consists of 48 steps and produces a 128-bit hash given a message of arbitrary finite size. MD5 is a more secure 64-step extension of MD4. Both MD4 and MD5 are vulnerable to practical collision attacks, yet it is still not realistic to invert them, i.e. to find a message given a hash. In 2007, the 39-step version of MD4 was inverted via reducing to SAT and applying a CDCL solver along with the so-called Dobbertin's constraints. As for MD5, in 2012 its 28-step version was inverted via a CDCL solver for one specified hash without adding any additional constraints. In this study, Cube-and-Conquer (a combination of CDCL and lookahead) is applied to invert step-reduced versions of MD4 and MD5. For this purpose, two algorithms are proposed. The first one generates inversion problems for MD4 by gradually modifying the Dobbertin's constraints. The second algorithm tries the cubing phase of Cube-and-Conquer with di
    

