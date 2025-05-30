# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ensuring User-side Fairness in Dynamic Recommender Systems.](http://arxiv.org/abs/2308.15651) | 本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。 |

# 详细

[^1]: 在动态推荐系统中确保用户侧公平性

    Ensuring User-side Fairness in Dynamic Recommender Systems. (arXiv:2308.15651v1 [cs.IR])

    [http://arxiv.org/abs/2308.15651](http://arxiv.org/abs/2308.15651)

    本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。

    

    用户侧群体公平性对现代推荐系统至关重要，它旨在减轻由敏感属性（如性别、种族或年龄）定义的用户群体之间的性能差异。我们发现这种差异往往会随着时间的推移而持续存在甚至增加。这需要在动态环境中有效解决用户侧公平性的方法，然而这在文献中很少被探讨。然而，用于确保用户侧公平性（即减少性能差异）的典型方法——公平约束重新排名，在动态设定中面临两个基本挑战：（1）基于排名的公平约束的非可微性，阻碍了端到端训练范式；（2）时间效率低下，阻碍了对用户偏好变化的快速适应。在本文中，我们提出了一种名为FADE的端到端框架，通过微调策略动态减轻性能差异。为了解决上述挑战，FADE提出了一种 fine-tuning 策略。

    User-side group fairness is crucial for modern recommender systems, as it aims to alleviate performance disparity between groups of users defined by sensitive attributes such as gender, race, or age. We find that the disparity tends to persist or even increase over time. This calls for effective ways to address user-side fairness in a dynamic environment, which has been infrequently explored in the literature. However, fairness-constrained re-ranking, a typical method to ensure user-side fairness (i.e., reducing performance disparity), faces two fundamental challenges in the dynamic setting: (1) non-differentiability of the ranking-based fairness constraint, which hinders the end-to-end training paradigm, and (2) time-inefficiency, which impedes quick adaptation to changes in user preferences. In this paper, we propose FAir Dynamic rEcommender (FADE), an end-to-end framework with fine-tuning strategy to dynamically alleviate performance disparity. To tackle the above challenges, FADE u
    

