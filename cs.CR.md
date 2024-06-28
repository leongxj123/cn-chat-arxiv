# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SoK: Can Trajectory Generation Combine Privacy and Utility?](https://arxiv.org/abs/2403.07218) | 本文提出了一个旨在设计保护隐私的轨迹发布方法的框架，特别强调了选择适当隐私单位的重要性。 |
| [^2] | [Data Reconstruction Attacks and Defenses: A Systematic Evaluation](https://arxiv.org/abs/2402.09478) | 本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。 |

# 详细

[^1]: SoK：轨迹生成是否能够兼顾隐私和实用性？

    SoK: Can Trajectory Generation Combine Privacy and Utility?

    [https://arxiv.org/abs/2403.07218](https://arxiv.org/abs/2403.07218)

    本文提出了一个旨在设计保护隐私的轨迹发布方法的框架，特别强调了选择适当隐私单位的重要性。

    

    虽然位置轨迹代表着供各种分析和基于位置的服务的宝贵数据来源，但它们可能泄漏敏感信息，如政治和宗教偏好。已经提出了不同ially private发布机制，允许在严格的隐私保证下进行分析。然而，传统的保护方案存在隐私和实用性的权衡限制，并容易受到相关性和重构攻击的威胁。合成轨迹数据生成和发布代表了保护算法的一个具有前景的替代方案。虽然最初的提议取得了显著的实用性，但未能提供严格的隐私保证。本文提出了一个框架，通过定义五个设计目标，特别强调选择适当的隐私单位的重要性，来设计一个保护隐私的轨迹发布方法。基于这一框架，我们简要讨论了现有的轨迹发布方法。

    arXiv:2403.07218v1 Announce Type: cross  Abstract: While location trajectories represent a valuable data source for analyses and location-based services, they can reveal sensitive information, such as political and religious preferences. Differentially private publication mechanisms have been proposed to allow for analyses under rigorous privacy guarantees. However, the traditional protection schemes suffer from a limiting privacy-utility trade-off and are vulnerable to correlation and reconstruction attacks. Synthetic trajectory data generation and release represent a promising alternative to protection algorithms. While initial proposals achieve remarkable utility, they fail to provide rigorous privacy guarantees. This paper proposes a framework for designing a privacy-preserving trajectory publication approach by defining five design goals, particularly stressing the importance of choosing an appropriate Unit of Privacy. Based on this framework, we briefly discuss the existing traje
    
[^2]: 数据重构攻击与防御：一个系统评估

    Data Reconstruction Attacks and Defenses: A Systematic Evaluation

    [https://arxiv.org/abs/2402.09478](https://arxiv.org/abs/2402.09478)

    本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。

    

    重构攻击和防御对于理解机器学习中的数据泄漏问题至关重要。然而，先前的工作主要集中在梯度反转攻击的经验观察上，缺乏理论基础，并且无法区分防御方法的有用性与攻击方法的计算限制。在这项工作中，我们提出了一种在联合学习环境中的强力重构攻击。该攻击可以重构中间特征，并与大部分先前的方法相比表现更好。在这种更强的攻击下，我们从理论和实证两方面全面调查了最常见的防御方法的效果。我们的研究结果表明，在各种防御机制中，如梯度剪辑、dropout、添加噪音、局部聚合等等，梯度修剪是对抗最先进攻击最有效的策略。

    arXiv:2402.09478v1 Announce Type: cross  Abstract: Reconstruction attacks and defenses are essential in understanding the data leakage problem in machine learning. However, prior work has centered around empirical observations of gradient inversion attacks, lacks theoretical groundings, and was unable to disentangle the usefulness of defending methods versus the computational limitation of attacking methods. In this work, we propose a strong reconstruction attack in the setting of federated learning. The attack reconstructs intermediate features and nicely integrates with and outperforms most of the previous methods. On this stronger attack, we thoroughly investigate both theoretically and empirically the effect of the most common defense methods. Our findings suggest that among various defense mechanisms, such as gradient clipping, dropout, additive noise, local aggregation, etc., gradient pruning emerges as the most effective strategy to defend against state-of-the-art attacks.
    

