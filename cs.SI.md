# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings.](http://arxiv.org/abs/2307.15702) | 强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。 |

# 详细

[^1]: 强大的最大环算法：一种集成偏好排序的新方法

    The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings. (arXiv:2307.15702v1 [cs.SI])

    [http://arxiv.org/abs/2307.15702](http://arxiv.org/abs/2307.15702)

    强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。

    

    我们提出了一种基于优化的方法，用于在每个决策者或选民对一对选择进行偏好表达的情况下集成偏好。挑战在于在一些冲突的投票情况下，尽可能与投票结果一致地得出一个排序。只有不包含环路的投票集合才是非冲突的，并且可以在选择之间引发一个部分顺序。我们的方法是基于这样一个观察：构成一个环路的投票集合可以被视为平局。然后，方法是从投票图中删除环路的并集，并根据剩余部分确定集成偏好。我们引入了强大的最大环路，它由一组环路的并集形成，删除它可以保证在引发的部分顺序中获得唯一结果。此外，它还包含在消除任何最大环路后剩下的所有集成偏好。与之相反的是，wel

    We present a new optimization-based method for aggregating preferences in settings where each decision maker, or voter, expresses preferences over pairs of alternatives. The challenge is to come up with a ranking that agrees as much as possible with the votes cast in cases when some of the votes conflict. Only a collection of votes that contains no cycles is non-conflicting and can induce a partial order over alternatives. Our approach is motivated by the observation that a collection of votes that form a cycle can be treated as ties. The method is then to remove unions of cycles of votes, or circulations, from the vote graph and determine aggregate preferences from the remainder.  We introduce the strong maximum circulation which is formed by a union of cycles, the removal of which guarantees a unique outcome in terms of the induced partial order. Furthermore, it contains all the aggregate preferences remaining following the elimination of any maximum circulation. In contrast, the wel
    

