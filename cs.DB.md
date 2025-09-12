# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inconsistency Handling in Prioritized Databases with Universal Constraints: Complexity Analysis and Links with Active Integrity Constraints.](http://arxiv.org/abs/2306.03523) | 本文研究解决了具有全局约束的不一致数据库的修复和查询问题，通过对称差分修复并指定首选修复行动，扩展了现有的最优修复概念，并且研究了修复概念的计算属性，同时澄清了与主动完整性约束框架中引入的修复概念之间的关系。 |

# 详细

[^1]: 具有全局约束的优先数据库中的不一致性处理：复杂度分析和与主动完整性约束的联系(arXiv:2306.03523v1 [cs.DB])

    Inconsistency Handling in Prioritized Databases with Universal Constraints: Complexity Analysis and Links with Active Integrity Constraints. (arXiv:2306.03523v1 [cs.DB])

    [http://arxiv.org/abs/2306.03523](http://arxiv.org/abs/2306.03523)

    本文研究解决了具有全局约束的不一致数据库的修复和查询问题，通过对称差分修复并指定首选修复行动，扩展了现有的最优修复概念，并且研究了修复概念的计算属性，同时澄清了与主动完整性约束框架中引入的修复概念之间的关系。

    

    本文重新审视了带有全局约束的不一致数据库的修复和查询问题。采用对称差分修复，即通过删除和添加事实来恢复一致性，并假设通过对（否定）事实的二元优先关系来指定首选修复行动。我们的第一个贡献是展示如何适当地将现有的最优修复概念（仅对基于事实删除的简单拒绝约束和修复定义）扩展到我们更丰富的设置中。接下来，我们研究了所得到的修复概念的计算属性，特别是修复检查和容忍不一致查询的数据复杂性。最后，我们澄清了优先数据库的最优修复与在主动完整性约束框架中引入的修复概念之间的关系。特别地，我们表明在我们的设置中的帕累托最优修复对应于 founded、grounded 和 just。

    This paper revisits the problem of repairing and querying inconsistent databases equipped with universal constraints. We adopt symmetric difference repairs, in which both deletions and additions of facts can be used to restore consistency, and suppose that preferred repair actions are specified via a binary priority relation over (negated) facts. Our first contribution is to show how existing notions of optimal repairs, defined for simpler denial constraints and repairs solely based on fact deletion, can be suitably extended to our richer setting. We next study the computational properties of the resulting repair notions, in particular, the data complexity of repair checking and inconsistency-tolerant query answering. Finally, we clarify the relationship between optimal repairs of prioritized databases and repair notions introduced in the framework of active integrity constraints. In particular, we show that Pareto-optimal repairs in our setting correspond to founded, grounded and just
    

