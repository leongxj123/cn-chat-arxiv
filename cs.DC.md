# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise.](http://arxiv.org/abs/2309.16105) | 本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。 |

# 详细

[^1]: 差分隐私安全乘法：在噪声中隐藏信息

    Differentially Private Secure Multiplication: Hiding Information in the Rubble of Noise. (arXiv:2309.16105v1 [cs.IT])

    [http://arxiv.org/abs/2309.16105](http://arxiv.org/abs/2309.16105)

    本文研究了在分布式计算中，允许信息泄漏和近似乘法的情况下，当诚实节点数量为少数时，差分隐私和准确性之间的权衡关系。

    

    我们考虑私密分布式多方乘法的问题。已经确认，Shamir秘密共享编码策略可以通过Ben Or，Goldwasser，Wigderson算法（“BGW算法”）在分布式计算中实现完美的信息理论隐私。然而，完美的隐私和准确性需要一个诚实的多数，即需要$N \geq 2t+1$个计算节点以确保对抗性节点的隐私。我们通过允许一定量的信息泄漏和近似乘法来研究在诚实节点数量为少数时的编码方案，即$N< 2t+1$。我们通过使用差分隐私而不是完美隐私来测量信息泄漏，并使用均方误差度量准确性，对$N < 2t+1$的情况下的隐私-准确性权衡进行了紧密的刻画。一个新颖的技术方面是复杂地控制信息泄漏的细节。

    We consider the problem of private distributed multi-party multiplication. It is well-established that Shamir secret-sharing coding strategies can enable perfect information-theoretic privacy in distributed computation via the celebrated algorithm of Ben Or, Goldwasser and Wigderson (the "BGW algorithm"). However, perfect privacy and accuracy require an honest majority, that is, $N \geq 2t+1$ compute nodes are required to ensure privacy against any $t$ colluding adversarial nodes. By allowing for some controlled amount of information leakage and approximate multiplication instead of exact multiplication, we study coding schemes for the setting where the number of honest nodes can be a minority, that is $N< 2t+1.$ We develop a tight characterization privacy-accuracy trade-off for cases where $N < 2t+1$ by measuring information leakage using {differential} privacy instead of perfect privacy, and using the mean squared error metric for accuracy. A novel technical aspect is an intricately 
    

