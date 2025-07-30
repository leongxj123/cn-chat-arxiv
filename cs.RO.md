# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Concurrent Multi-Robot Coverage Path Planning](https://arxiv.org/abs/2403.10460) | 提出了一种非地平线的集中式算法，实现了在线多机器人覆盖路径规划中的并发规划和执行。 |

# 详细

[^1]: 在线并发多机器人覆盖路径规划

    Online Concurrent Multi-Robot Coverage Path Planning

    [https://arxiv.org/abs/2403.10460](https://arxiv.org/abs/2403.10460)

    提出了一种非地平线的集中式算法，实现了在线多机器人覆盖路径规划中的并发规划和执行。

    

    近期，集中式逐步地平线在线多机器人覆盖路径规划算法展现出在彻底探索拥有大量机器人的大型、复杂、未知工作空间方面的出色可伸缩性。在一个时间段内，路径规划和路径执行交替进行，即当为没有路径的机器人进行路径规划时，具有未完成路径的机器人不执行，反之亦然。为此，我们提出了一个非基于地平线的集中式算法。该算法随时为没有路径的机器人子集（即已达到其先前分配目标的机器人）规划路径，而其余机器人执行其未完成的路径，从而实现并发规划和执行。我们正式证明了该提议的...

    arXiv:2403.10460v1 Announce Type: cross  Abstract: Recently, centralized receding horizon online multi-robot coverage path planning algorithms have shown remarkable scalability in thoroughly exploring large, complex, unknown workspaces with many robots. In a horizon, the path planning and the path execution interleave, meaning when the path planning occurs for robots with no paths, the robots with outstanding paths do not execute, and subsequently, when the robots with new or outstanding paths execute to reach respective goals, path planning does not occur for those robots yet to get new paths, leading to wastage of both the robotic and the computation resources. As a remedy, we propose a centralized algorithm that is not horizon-based. It plans paths at any time for a subset of robots with no paths, i.e., who have reached their previously assigned goals, while the rest execute their outstanding paths, thereby enabling concurrent planning and execution. We formally prove that the propo
    

