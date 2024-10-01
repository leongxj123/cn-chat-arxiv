# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regulation of Algorithmic Collusion.](http://arxiv.org/abs/2401.15794) | 本文提供了算法定价中的算法合谋监管的定义，允许监管机构通过对数据的统计测试来审计算法，良好算法通过增加数据来通过审计，而达到高于竞争价格的算法则无法通过审计。 |
| [^2] | [Active Inverse Learning in Stackelberg Trajectory Games.](http://arxiv.org/abs/2308.08017) | 这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。 |
| [^3] | [Balanced Donor Coordination.](http://arxiv.org/abs/2305.10286) | 这篇论文提出了一种既尊重个人意愿又高效的捐赠机制，通过以高效的方式分配每个捐赠者的贡献，没有任何捐助人有动机重新分配他们的捐款。 |

# 详细

[^1]: 算法定价中的算法合谋监管

    Regulation of Algorithmic Collusion. (arXiv:2401.15794v1 [cs.GT])

    [http://arxiv.org/abs/2401.15794](http://arxiv.org/abs/2401.15794)

    本文提供了算法定价中的算法合谋监管的定义，允许监管机构通过对数据的统计测试来审计算法，良好算法通过增加数据来通过审计，而达到高于竞争价格的算法则无法通过审计。

    

    考虑到在竞争市场中，卖方使用算法根据他们收集到的数据调整价格。在这种情况下，算法可能得出高于竞争价格的价格，从而使卖方受益，而损害消费者（即市场中的买方）。本文为定价算法提供了合理的算法非合谋的定义。该定义允许监管机构通过对所收集数据的统计测试来对算法进行实证审计。良好的算法，即在市场条件下近似优化价格的算法，可以通过增加足够数据来通过审计。相反，达到高于竞争价格的算法无法通过审计。该定义允许卖方拥有与供需相互关联且可能影响良好算法所使用的价格的有用辅助信息。本文还分析了统计互补性。

    Consider sellers in a competitive market that use algorithms to adapt their prices from data that they collect. In such a context it is plausible that algorithms could arrive at prices that are higher than the competitive prices and this may benefit sellers at the expense of consumers (i.e., the buyers in the market). This paper gives a definition of plausible algorithmic non-collusion for pricing algorithms. The definition allows a regulator to empirically audit algorithms by applying a statistical test to the data that they collect. Algorithms that are good, i.e., approximately optimize prices to market conditions, can be augmented to contain the data sufficient to pass the audit. Algorithms that have colluded on, e.g., supra-competitive prices cannot pass the audit. The definition allows sellers to possess useful side information that may be correlated with supply and demand and could affect the prices used by good algorithms. The paper provides an analysis of the statistical comple
    
[^2]: Stackelberg轨迹博弈中的主动逆向学习

    Active Inverse Learning in Stackelberg Trajectory Games. (arXiv:2308.08017v1 [cs.GT])

    [http://arxiv.org/abs/2308.08017](http://arxiv.org/abs/2308.08017)

    这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。

    

    博弈论的逆向学习是从玩家的行为中推断出他们的目标的问题。我们在一个Stackelberg博弈中，通过每个玩家的动态系统轨迹来定义一个逆向学习问题，其中包括一个领导者和一个跟随者。我们提出了一种主动逆向学习方法，让领导者推断出一个有限候选集中描述跟随者目标函数的假设。与现有方法使用被动观察到的轨迹不同，所提出的方法主动地最大化不同假设下跟随者轨迹的差异，加速领导者的推断过程。我们在一个递进的重复轨迹博弈中展示了所提出的方法。与均匀随机输入相比，所提供的方法加速了概率收敛到条件于跟随者轨迹的不同假设上的收敛速度。

    Game-theoretic inverse learning is the problem of inferring the players' objectives from their actions. We formulate an inverse learning problem in a Stackelberg game between a leader and a follower, where each player's action is the trajectory of a dynamical system. We propose an active inverse learning method for the leader to infer which hypothesis among a finite set of candidates describes the follower's objective function. Instead of using passively observed trajectories like existing methods, the proposed method actively maximizes the differences in the follower's trajectories under different hypotheses to accelerate the leader's inference. We demonstrate the proposed method in a receding-horizon repeated trajectory game. Compared with uniformly random inputs, the leader inputs provided by the proposed method accelerate the convergence of the probability of different hypotheses conditioned on the follower's trajectory by orders of magnitude.
    
[^3]: 平衡捐赠者协调

    Balanced Donor Coordination. (arXiv:2305.10286v1 [econ.TH])

    [http://arxiv.org/abs/2305.10286](http://arxiv.org/abs/2305.10286)

    这篇论文提出了一种既尊重个人意愿又高效的捐赠机制，通过以高效的方式分配每个捐赠者的贡献，没有任何捐助人有动机重新分配他们的捐款。

    

    慈善通常由个人捐赠者或中央组织（如政府或市政机构）进行，他们收集个人捐款并将它们分配给一组慈善机构。一方面，个人慈善尊重捐赠者的意愿，但由于缺乏协调而可能效率低下。另一方面，集中慈善可能更有效，但可能忽略个人捐赠者的意愿。我们提出了一种机制，它结合了两种方法的优点，通过以高效的方式分配每个捐赠者的贡献，没有任何捐助人有动机重新分配他们的捐款。在假设Leontief效用函数（即每个捐赠者均希望最大化所有慈善机构的最小贡献的个人加权之和）的情况下，我们的机制是组策略正确、偏好单调、贡献单调、最大化纳什福利并可以计算。

    Charity is typically done either by individual donors, who donate money to the charities that they support, or by centralized organizations such as governments or municipalities, which collect the individual contributions and distribute them among a set of charities. On the one hand, individual charity respects the will of the donors but may be inefficient due to a lack of coordination. On the other hand, centralized charity is potentially more efficient but may ignore the will of individual donors. We present a mechanism that combines the advantages of both methods by distributing the contribution of each donor in an efficient way such that no subset of donors has an incentive to redistribute their donations. Assuming Leontief utilities (i.e., each donor is interested in maximizing an individually weighted minimum of all contributions across the charities), our mechanism is group-strategyproof, preference-monotonic, contribution-monotonic, maximizes Nash welfare, and can be computed u
    

