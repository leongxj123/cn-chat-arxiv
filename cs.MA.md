# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Locally Differentially Private Distributed Online Learning with Guaranteed Optimality.](http://arxiv.org/abs/2306.14094) | 本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。 |

# 详细

[^1]: 具有保证最优性的局部差分隐私分布式在线学习

    Locally Differentially Private Distributed Online Learning with Guaranteed Optimality. (arXiv:2306.14094v1 [cs.LG])

    [http://arxiv.org/abs/2306.14094](http://arxiv.org/abs/2306.14094)

    本文提出了一种具有保证最优性的方法，可以在分布式在线学习中同时保证差分隐私和学习准确性。

    

    分布式在线学习由于其处理大规模数据集和流数据的能力而受到越来越多的关注。为了解决隐私保护问题，已经提出了许多个人私密分布式在线学习算法，大多数基于差分隐私，差分隐私已成为隐私保护的“黄金标准”。然而，这些算法常常面临为了隐私保护而牺牲学习准确性的困境。本文利用在线学习的独特特征，提出了一种方法来解决这一困境，并确保分布式在线学习中的差分隐私和学习准确性。具体而言，该方法在确保预期瞬时遗憾程度逐渐减小的同时，还能保证有限的累积隐私预算，即使在无限时间范围内。为了应对完全分布式环境，我们采用本地差分隐私框架，避免了对全局数据的依赖。

    Distributed online learning is gaining increased traction due to its unique ability to process large-scale datasets and streaming data. To address the growing public awareness and concern on privacy protection, plenty of private distributed online learning algorithms have been proposed, mostly based on differential privacy which has emerged as the ``gold standard" for privacy protection. However, these algorithms often face the dilemma of trading learning accuracy for privacy. By exploiting the unique characteristics of online learning, this paper proposes an approach that tackles the dilemma and ensures both differential privacy and learning accuracy in distributed online learning. More specifically, while ensuring a diminishing expected instantaneous regret, the approach can simultaneously ensure a finite cumulative privacy budget, even on the infinite time horizon. To cater for the fully distributed setting, we adopt the local differential-privacy framework which avoids the reliance
    

