# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Understanding of Why Users Share Hate Speech on Social Media.](http://arxiv.org/abs/2310.15772) | 本文研究了用户为何分享社交媒体上的仇恨言论，提出了一个因果分析框架，通过消除数据偏差和模拟用户脆弱性来揭示影响用户分享行为的因素。 |
| [^2] | [The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings.](http://arxiv.org/abs/2307.15702) | 强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。 |

# 详细

[^1]: 用户在社交媒体上分享仇恨言论的因果理解

    Causal Understanding of Why Users Share Hate Speech on Social Media. (arXiv:2310.15772v1 [cs.SI])

    [http://arxiv.org/abs/2310.15772](http://arxiv.org/abs/2310.15772)

    本文研究了用户为何分享社交媒体上的仇恨言论，提出了一个因果分析框架，通过消除数据偏差和模拟用户脆弱性来揭示影响用户分享行为的因素。

    

    社交媒体上的仇恨言论威胁到个人的心理和身体健康，并且进一步导致现实中的暴力事件。仇恨言论传播背后的重要驱动因素是转发，但是人们很少了解为什么用户会转发仇恨言论。本文提供了一个全面、因果分析的用户属性框架，研究用户为何分享仇恨言论。然而，在从社交媒体数据中进行因果推断时存在一些挑战，因为这类数据很可能存在选择偏差，并且用户对仇恨言论的脆弱性存在混淆。我们开发了一个新颖的三步因果框架：（1）我们通过逆向倾向评分来消除观察性社交媒体数据的偏差。（2）我们使用消除偏差的倾向评分来模拟用户对仇恨言论的潜在脆弱性作为潜在嵌入。（3）我们建立了用户属性对用户分享仇恨言论概率的因果效应模型。

    Hate speech on social media threatens the mental and physical well-being of individuals and is further responsible for real-world violence. An important driver behind the spread of hate speech and thus why hateful posts can go viral are reshares, yet little is known about why users reshare hate speech. In this paper, we present a comprehensive, causal analysis of the user attributes that make users reshare hate speech. However, causal inference from observational social media data is challenging, because such data likely suffer from selection bias, and there is further confounding due to differences in the vulnerability of users to hate speech. We develop a novel, three-step causal framework: (1) We debias the observational social media data by applying inverse propensity scoring. (2) We use the debiased propensity scores to model the latent vulnerability of users to hate speech as a latent embedding. (3) We model the causal effects of user attributes on users' probability of sharing h
    
[^2]: 强大的最大环算法：一种集成偏好排序的新方法

    The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings. (arXiv:2307.15702v1 [cs.SI])

    [http://arxiv.org/abs/2307.15702](http://arxiv.org/abs/2307.15702)

    强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。

    

    我们提出了一种基于优化的方法，用于在每个决策者或选民对一对选择进行偏好表达的情况下集成偏好。挑战在于在一些冲突的投票情况下，尽可能与投票结果一致地得出一个排序。只有不包含环路的投票集合才是非冲突的，并且可以在选择之间引发一个部分顺序。我们的方法是基于这样一个观察：构成一个环路的投票集合可以被视为平局。然后，方法是从投票图中删除环路的并集，并根据剩余部分确定集成偏好。我们引入了强大的最大环路，它由一组环路的并集形成，删除它可以保证在引发的部分顺序中获得唯一结果。此外，它还包含在消除任何最大环路后剩下的所有集成偏好。与之相反的是，wel

    We present a new optimization-based method for aggregating preferences in settings where each decision maker, or voter, expresses preferences over pairs of alternatives. The challenge is to come up with a ranking that agrees as much as possible with the votes cast in cases when some of the votes conflict. Only a collection of votes that contains no cycles is non-conflicting and can induce a partial order over alternatives. Our approach is motivated by the observation that a collection of votes that form a cycle can be treated as ties. The method is then to remove unions of cycles of votes, or circulations, from the vote graph and determine aggregate preferences from the remainder.  We introduce the strong maximum circulation which is formed by a union of cycles, the removal of which guarantees a unique outcome in terms of the induced partial order. Furthermore, it contains all the aggregate preferences remaining following the elimination of any maximum circulation. In contrast, the wel
    

