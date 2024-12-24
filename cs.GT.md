# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Manipulate under Limited Information.](http://arxiv.org/abs/2401.16412) | 本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。 |
| [^2] | [A Unified Approach to Second and Third Degree Price Discrimination.](http://arxiv.org/abs/2401.12366) | 本文提出了一个统一方法来分析二、三度价格歧视的福利影响。通过选择特定的分割方式，可以实现不同的福利结果，并且分析了分割对消费者剩余和价格的影响。同时，提出了一个高效算法来计算分割。 |

# 详细

[^1]: 学习在有限信息下进行操纵

    Learning to Manipulate under Limited Information. (arXiv:2401.16412v1 [cs.AI])

    [http://arxiv.org/abs/2401.16412](http://arxiv.org/abs/2401.16412)

    本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。

    

    根据社会选择理论的经典结果，任何合理的偏好投票方法有时会给个体提供报告不真实偏好的激励。对于比较投票方法来说，不同投票方法在多大程度上更或者更少抵抗这种策略性操纵已成为一个关键考虑因素。在这里，我们通过神经网络在不同规模下对限制信息下学习如何利用给定投票方法进行操纵的成功程度来衡量操纵的抵抗力。我们训练了将近40,000个不同规模的神经网络来对抗8种不同的投票方法，在6种限制信息情况下，进行包含5-21名选民和3-6名候选人的委员会规模选举的操纵。我们发现，一些投票方法，如Borda方法，在有限信息下可以被神经网络高度操纵，而其他方法，如Instant Runoff方法，虽然被一个理想的操纵者利润化操纵，但在有限信息下不会受到操纵。

    By classic results in social choice theory, any reasonable preferential voting method sometimes gives individuals an incentive to report an insincere preference. The extent to which different voting methods are more or less resistant to such strategic manipulation has become a key consideration for comparing voting methods. Here we measure resistance to manipulation by whether neural networks of varying sizes can learn to profitably manipulate a given voting method in expectation, given different types of limited information about how other voters will vote. We trained nearly 40,000 neural networks of 26 sizes to manipulate against 8 different voting methods, under 6 types of limited information, in committee-sized elections with 5-21 voters and 3-6 candidates. We find that some voting methods, such as Borda, are highly manipulable by networks with limited information, while others, such as Instant Runoff, are not, despite being quite profitably manipulated by an ideal manipulator with
    
[^2]: 二、三度价格歧视的统一方法

    A Unified Approach to Second and Third Degree Price Discrimination. (arXiv:2401.12366v1 [econ.TH])

    [http://arxiv.org/abs/2401.12366](http://arxiv.org/abs/2401.12366)

    本文提出了一个统一方法来分析二、三度价格歧视的福利影响。通过选择特定的分割方式，可以实现不同的福利结果，并且分析了分割对消费者剩余和价格的影响。同时，提出了一个高效算法来计算分割。

    

    本文分析了一个能够对多产品市场进行分割并在每个分割中提供差异化价格菜单的垄断者的福利影响。我们描述了一组极端分布，通过从这些分布内选择分割，可以达到所有可行的福利结果。这个分布族是对多商品市场中消费者最大化价值分布的解决方案产生的。根据这些结果，我们分析了分割对消费者剩余和价格的影响，包括存在使所有消费者受益的分割的条件。最后，我们提出了一个高效的算法来计算分割。

    We analyze the welfare impact of a monopolist able to segment a multiproduct market and offer differentiated price menus within each segment. We characterize a family of extremal distributions such that all achievable welfare outcomes can be reached by selecting segments from within these distributions. This family of distributions arises as the solution to the consumer maximizing distribution of values for multigood markets. With these results, we analyze the effect of segmentation on consumer surplus and prices in both interior and extremal markets, including conditions under which there exists a segmentation benefiting all consumers. Finally, we present an efficient algorithm for computing segmentations.
    

