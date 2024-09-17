# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LTP-MMF: Towards Long-term Provider Max-min Fairness Under Recommendation Feedback Loops.](http://arxiv.org/abs/2308.05902) | 该论文提出了一种在线排序模型，名为长期供应商MMF，以解决推荐反馈循环下的长期供应商最大最小公平性的挑战。 |

# 详细

[^1]: LTP-MMF: 面向推荐反馈循环下的长期供应商最大最小公平性

    LTP-MMF: Towards Long-term Provider Max-min Fairness Under Recommendation Feedback Loops. (arXiv:2308.05902v1 [cs.IR])

    [http://arxiv.org/abs/2308.05902](http://arxiv.org/abs/2308.05902)

    该论文提出了一种在线排序模型，名为长期供应商MMF，以解决推荐反馈循环下的长期供应商最大最小公平性的挑战。

    

    多利益相关者推荐系统涉及各种角色，如用户、供应商。先前的研究指出，最大最小公平性（MMF）是支持弱供应商的更好指标。然而，考虑到MMF时，这些角色的特征或参数会随时间变化，如何确保长期供应商MMF已经成为一个重要挑战。我们观察到，推荐反馈循环（RFL）会对供应商的MMF产生重大影响。RFL意味着推荐系统只能从用户那里接收到已公开物品的反馈，并根据这些反馈增量更新推荐模型。在利用反馈时，推荐模型将把未公开的物品视为负面样本。这样，尾部供应商将无法被曝光，其物品将始终被视为负面样本。在RFL中，这种现象会越来越严重。为了缓解这个问题，本文提出了一个名为长期供应商MMF的在线排序模型。

    Multi-stakeholder recommender systems involve various roles, such as users, providers. Previous work pointed out that max-min fairness (MMF) is a better metric to support weak providers. However, when considering MMF, the features or parameters of these roles vary over time, how to ensure long-term provider MMF has become a significant challenge. We observed that recommendation feedback loops (named RFL) will influence the provider MMF greatly in the long term. RFL means that recommender system can only receive feedback on exposed items from users and update recommender models incrementally based on this feedback. When utilizing the feedback, the recommender model will regard unexposed item as negative. In this way, tail provider will not get the opportunity to be exposed, and its items will always be considered as negative samples. Such phenomenons will become more and more serious in RFL. To alleviate the problem, this paper proposes an online ranking model named Long-Term Provider M
    

