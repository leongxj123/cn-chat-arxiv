# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning in Repeated Multi-Unit Pay-As-Bid Auctions.](http://arxiv.org/abs/2307.15193) | 本论文研究了在重复的多单位付费拍卖中学习如何出价的问题。通过在离线设置中优化出价向量，并利用多项式时间动态规划方案，设计了具有多项式时间和空间复杂度的在线学习算法。 |

# 详细

[^1]: 在重复的多单位付费拍卖中学习

    Learning in Repeated Multi-Unit Pay-As-Bid Auctions. (arXiv:2307.15193v1 [cs.GT])

    [http://arxiv.org/abs/2307.15193](http://arxiv.org/abs/2307.15193)

    本论文研究了在重复的多单位付费拍卖中学习如何出价的问题。通过在离线设置中优化出价向量，并利用多项式时间动态规划方案，设计了具有多项式时间和空间复杂度的在线学习算法。

    

    受碳排放交易方案、国债拍卖和采购拍卖的启发，这些都涉及拍卖同质的多个单位，我们考虑了如何在重复的多单位付费拍卖中学习如何出价的问题。在每个拍卖中，大量（相同的）物品将被分配给最高的出价，每个中标价等于出价本身。由于行动空间的组合性质，学习如何在付费拍卖中出价是具有挑战性的。为了克服这个挑战，我们关注离线设置，其中投标人通过只能访问其他投标人过去提交的出价来优化他们的出价向量。我们证明了离线问题的最优解可以使用多项式时间动态规划（DP）方案来获得。我们利用DP方案的结构，设计了具有多项式时间和空间复杂度的在线学习算法。

    Motivated by Carbon Emissions Trading Schemes, Treasury Auctions, and Procurement Auctions, which all involve the auctioning of homogeneous multiple units, we consider the problem of learning how to bid in repeated multi-unit pay-as-bid auctions. In each of these auctions, a large number of (identical) items are to be allocated to the largest submitted bids, where the price of each of the winning bids is equal to the bid itself. The problem of learning how to bid in pay-as-bid auctions is challenging due to the combinatorial nature of the action space. We overcome this challenge by focusing on the offline setting, where the bidder optimizes their vector of bids while only having access to the past submitted bids by other bidders. We show that the optimal solution to the offline problem can be obtained using a polynomial time dynamic programming (DP) scheme. We leverage the structure of the DP scheme to design online learning algorithms with polynomial time and space complexity under fu
    

