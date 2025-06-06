# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications.](http://arxiv.org/abs/2401.04003) | 该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。 |

# 详细

[^1]: 多机器人在层次化时间逻辑规范下的任务分配和规划

    Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications. (arXiv:2401.04003v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2401.04003](http://arxiv.org/abs/2401.04003)

    该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。

    

    过去关于机器人规划与时间逻辑规范的研究，特别是线性时间逻辑（LTL），主要是基于针对个体或群体机器人的单一公式。但随着任务复杂性的增加，LTL公式不可避免地变得冗长，使解释和规范生成变得复杂，同时还对规划器的计算能力造成压力。通过利用任务的内在结构，我们引入了一种层次化结构到具有语法和语义需求的LTL规范中，并证明它们比扁平规范更具表达能力。其次，我们采用基于搜索的方法来综合多机器人系统的计划，实现同时的任务分配和规划。搜索空间由松散相互连接的子空间近似表示，每个子空间对应一个LTL规范。搜索主要受限于单个子空间，根据特定条件转移到另一个子空间。

    Past research into robotic planning with temporal logic specifications, notably Linear Temporal Logic (LTL), was largely based on singular formulas for individual or groups of robots. But with increasing task complexity, LTL formulas unavoidably grow lengthy, complicating interpretation and specification generation, and straining the computational capacities of the planners. By leveraging the intrinsic structure of tasks, we introduced a hierarchical structure to LTL specifications with requirements on syntax and semantics, and proved that they are more expressive than their flat counterparts. Second, we employ a search-based approach to synthesize plans for a multi-robot system, accomplishing simultaneous task allocation and planning. The search space is approximated by loosely interconnected sub-spaces, with each sub-space corresponding to one LTL specification. The search is predominantly confined to a single sub-space, transitioning to another sub-space under certain conditions, de
    

