# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximizing Seaweed Growth on Autonomous Farms: A Dynamic Programming Approach for Underactuated Systems Navigating on Uncertain Ocean Currents.](http://arxiv.org/abs/2307.01916) | 设计了一种基于动态规划的方法，用于在不确定的海洋洋流中最大化海藻生长，通过利用非线性时变的洋流实现高生长区域的探测。 |

# 详细

[^1]: 在不确定的海洋洋流中进行动力编程的无效系统上的自主农场上的海藻生长的最大化方法

    Maximizing Seaweed Growth on Autonomous Farms: A Dynamic Programming Approach for Underactuated Systems Navigating on Uncertain Ocean Currents. (arXiv:2307.01916v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2307.01916](http://arxiv.org/abs/2307.01916)

    设计了一种基于动态规划的方法，用于在不确定的海洋洋流中最大化海藻生长，通过利用非线性时变的洋流实现高生长区域的探测。

    

    海藻生物量在气候减缓方面具有重要潜力，但需要大规模的自主开放式海洋农场来充分利用。这些农场通常具有低推进力，并受到海洋洋流的重大影响。我们希望设计一个控制器，通过利用非线性时变的海洋洋流来达到高生长区域，从而在几个月内最大化海藻生长。复杂的动力学和无效性使得即使知道洋流情况，这也是具有挑战性的。当只有短期不完善的预测且不确定性逐渐增大时，情况变得更加困难。我们提出了一种基于动态规划的方法，可以在已知真实洋流情况时有效地求解最优生长值函数。此外，我们还提出了三个扩展，即在现实中只知道预测的情况下：（1）我们方法得到的值函数可以作为反馈策略，以获得所有状态和时间的最佳生长控制，实现闭环控制的等价性。

    Seaweed biomass offers significant potential for climate mitigation, but large-scale, autonomous open-ocean farms are required to fully exploit it. Such farms typically have low propulsion and are heavily influenced by ocean currents. We want to design a controller that maximizes seaweed growth over months by taking advantage of the non-linear time-varying ocean currents for reaching high-growth regions. The complex dynamics and underactuation make this challenging even when the currents are known. This is even harder when only short-term imperfect forecasts with increasing uncertainty are available. We propose a dynamic programming-based method to efficiently solve for the optimal growth value function when true currents are known. We additionally present three extensions when as in reality only forecasts are known: (1) our methods resulting value function can be used as feedback policy to obtain the growth-optimal control for all states and times, allowing closed-loop control equival
    

