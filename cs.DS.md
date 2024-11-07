# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Breaking the Metric Voting Distortion Barrier.](http://arxiv.org/abs/2306.17838) | 这篇论文研究了社会选择中的度量扭曲问题，提出了一个新的投票规则，该规则可以选择距离选民平均距离较小的候选人。之前的研究发现确定性投票规则的度量扭曲限制为3，但在无限制的情况下，我们对该问题仍然了解有限。 |

# 详细

[^1]: 打破度量投票扭曲的障碍

    Breaking the Metric Voting Distortion Barrier. (arXiv:2306.17838v1 [cs.GT])

    [http://arxiv.org/abs/2306.17838](http://arxiv.org/abs/2306.17838)

    这篇论文研究了社会选择中的度量扭曲问题，提出了一个新的投票规则，该规则可以选择距离选民平均距离较小的候选人。之前的研究发现确定性投票规则的度量扭曲限制为3，但在无限制的情况下，我们对该问题仍然了解有限。

    

    我们考虑社会选择中度量扭曲的经典问题。假设我们有一个选举，有n名选民和m名候选人，他们位于一个共享的度量空间中。我们希望设计一个投票规则，选择一个平均距离选民较小的候选人。然而，我们不能直接获得度量空间中的距离信息，每个选民只能给出候选人的排序列表。我们能否设计一条规则，无论选举实例和底层度量空间如何，都能选择出一个与真正最优解的代价只相差一个小因子（称为扭曲度）的候选人？许多研究的成果将确定性投票规则的度量扭曲限制为3，这是确定性规则和许多其他投票规则类别的最佳结果。然而，在没有任何限制的情况下，我们对该问题仍然了解有限：尽管最佳下界已经降低到2.112，但现有规则的扭曲度仍然相对较高。

    We consider the following well studied problem of metric distortion in social choice. Suppose we have an election with $n$ voters and $m$ candidates who lie in a shared metric space. We would like to design a voting rule that chooses a candidate whose average distance to the voters is small. However, instead of having direct access to the distances in the metric space, each voter gives us a ranked list of the candidates in order of distance. Can we design a rule that regardless of the election instance and underlying metric space, chooses a candidate whose cost differs from the true optimum by only a small factor (known as the distortion)?  A long line of work culminated in finding deterministic voting rules with metric distortion $3$, which is the best possible for deterministic rules and many other classes of voting rules. However, without any restrictions, there is still a significant gap in our understanding: Even though the best lower bound is substantially lower at $2.112$, the b
    

