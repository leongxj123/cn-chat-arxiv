# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm.](http://arxiv.org/abs/2401.08150) | 本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。 |
| [^2] | [Backdoor Defense with Non-Adversarial Backdoor.](http://arxiv.org/abs/2307.15539) | 提出了一种非对抗性后门防御框架，通过在被污染样本中注入非对抗性后门，当触发时可以抑制攻击者对污染数据的后门攻击，同时保持对干净数据的影响有限。 |

# 详细

[^1]: 差分隐私切片逆回归: 极小极大性和算法

    Differentially Private Sliced Inverse Regression: Minimax Optimality and Algorithm. (arXiv:2401.08150v1 [stat.ML])

    [http://arxiv.org/abs/2401.08150](http://arxiv.org/abs/2401.08150)

    本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法，并在低维和高维设置下建立了不同ially private 切片逆回归的下界。通过仿真和真实数据分析验证了这些算法的有效性。

    

    随着数据驱动应用的普及，隐私保护已成为高维数据分析中的一个关键问题。切片逆回归是一种广泛应用的统计技术，通过降低协变量的维度，同时保持足够的统计信息。本文提出了针对充足维度减少中的隐私问题的最佳差分隐私算法。我们在低维和高维设置下建立了不同ially private 切片逆回归的下界。此外，我们设计了差分隐私算法，实现了极小极大下界的要求，并在降维空间中同时保护隐私和保存重要信息的有效性。通过一系列的仿真实验和真实数据分析，我们证明了这些差分隐私算法的有效性。

    Privacy preservation has become a critical concern in high-dimensional data analysis due to the growing prevalence of data-driven applications. Proposed by Li (1991), sliced inverse regression has emerged as a widely utilized statistical technique for reducing covariate dimensionality while maintaining sufficient statistical information. In this paper, we propose optimally differentially private algorithms specifically designed to address privacy concerns in the context of sufficient dimension reduction. We proceed to establish lower bounds for differentially private sliced inverse regression in both the low and high-dimensional settings. Moreover, we develop differentially private algorithms that achieve the minimax lower bounds up to logarithmic factors. Through a combination of simulations and real data analysis, we illustrate the efficacy of these differentially private algorithms in safeguarding privacy while preserving vital information within the reduced dimension space. As a na
    
[^2]: 非对抗性后门防御

    Backdoor Defense with Non-Adversarial Backdoor. (arXiv:2307.15539v1 [cs.LG])

    [http://arxiv.org/abs/2307.15539](http://arxiv.org/abs/2307.15539)

    提出了一种非对抗性后门防御框架，通过在被污染样本中注入非对抗性后门，当触发时可以抑制攻击者对污染数据的后门攻击，同时保持对干净数据的影响有限。

    

    深度神经网络（DNNs）容易受到后门攻击的影响，这种攻击并不会影响网络对干净数据的性能，但一旦添加触发模式，就会操纵网络行为。现有的防御方法大大降低了攻击成功率，但它们在干净数据上的预测准确性仍然远远落后于干净模型。受后门攻击的隐蔽性和有效性的启发，我们提出了一个简单但非常有效的防御框架，该框架注入了针对被污染样本的非对抗性后门。按照后门攻击的一般步骤，我们检测一小组可疑样本，然后对它们应用毒化策略。一旦触发，非对抗性后门抑制了攻击者对污染数据的后门攻击，但对干净数据的影响有限。防御可以在数据预处理期间进行，而不需要对标准的端到端训练流程进行任何修改。

    Deep neural networks (DNNs) are vulnerable to backdoor attack, which does not affect the network's performance on clean data but would manipulate the network behavior once a trigger pattern is added. Existing defense methods have greatly reduced attack success rate, but their prediction accuracy on clean data still lags behind a clean model by a large margin. Inspired by the stealthiness and effectiveness of backdoor attack, we propose a simple but highly effective defense framework which injects non-adversarial backdoors targeting poisoned samples. Following the general steps in backdoor attack, we detect a small set of suspected samples and then apply a poisoning strategy to them. The non-adversarial backdoor, once triggered, suppresses the attacker's backdoor on poisoned data, but has limited influence on clean data. The defense can be carried out during data preprocessing, without any modification to the standard end-to-end training pipeline. We conduct extensive experiments on mul
    

