# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimally Modifying a Markov Game to Achieve Any Nash Equilibrium and Value.](http://arxiv.org/abs/2311.00582) | 该论文研究了游戏修改问题，提出了一种最小修改马尔可夫博弈的方法，使得目标策略配置成为唯一的Nash均衡并具有特定价值范围，同时最小化修改成本。 |
| [^2] | [Performative Prediction with Neural Networks.](http://arxiv.org/abs/2304.06879) | 本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。 |

# 详细

[^1]: 最小修改马尔可夫博弈以实现任意Nash均衡和价值

    Minimally Modifying a Markov Game to Achieve Any Nash Equilibrium and Value. (arXiv:2311.00582v1 [cs.GT])

    [http://arxiv.org/abs/2311.00582](http://arxiv.org/abs/2311.00582)

    该论文研究了游戏修改问题，提出了一种最小修改马尔可夫博弈的方法，使得目标策略配置成为唯一的Nash均衡并具有特定价值范围，同时最小化修改成本。

    

    我们研究了游戏修改问题，其中一位善意的游戏设计者或恶意的对手修改了一个零和马尔可夫博弈的奖励函数，以便一个目标确定性或随机的策略配置成为唯一的马尔可夫完美Nash均衡，并且在目标范围内具有价值，以最小化修改成本。我们表征了能够安装为某个游戏的唯一均衡的策略配置的集合，并建立了成功安装的充分和必要条件。我们提出了一种高效的算法，该算法通过解一个带有线性约束的凸优化问题，然后进行随机扰动，来获得一个成本近乎最优的修改计划。

    We study the game modification problem, where a benevolent game designer or a malevolent adversary modifies the reward function of a zero-sum Markov game so that a target deterministic or stochastic policy profile becomes the unique Markov perfect Nash equilibrium and has a value within a target range, in a way that minimizes the modification cost. We characterize the set of policy profiles that can be installed as the unique equilibrium of some game, and establish sufficient and necessary conditions for successful installation. We propose an efficient algorithm, which solves a convex optimization problem with linear constraints and then performs random perturbation, to obtain a modification plan with a near-optimal cost.
    
[^2]: 神经网络下的执行预测

    Performative Prediction with Neural Networks. (arXiv:2304.06879v1 [cs.LG])

    [http://arxiv.org/abs/2304.06879](http://arxiv.org/abs/2304.06879)

    本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。

    

    执行预测是一种学习模型并影响其预测数据的框架。本文旨在找到分类器，使其具有执行稳定性，即适用于其产生的数据分布的最佳分类器。在使用重复风险最小化方法找到具有执行稳定性的分类器的标准收敛结果中，假设数据分布对于模型参数是可Lipschitz连续的。在这种情况下，损失必须对这些参数强凸和平滑；否则，该方法将在某些问题上发散。然而本文则假设数据分布是相对于模型的预测值可Lipschitz连续的，这是执行系统的更加自然的假设。结果，我们能够显著放宽对损失函数的假设要求。作为一个说明，我们介绍了一种建模真实数据分布的重采样过程，并使用其来实证执行稳定性相对于其他目标的效益。

    Performative prediction is a framework for learning models that influence the data they intend to predict. We focus on finding classifiers that are performatively stable, i.e. optimal for the data distribution they induce. Standard convergence results for finding a performatively stable classifier with the method of repeated risk minimization assume that the data distribution is Lipschitz continuous to the model's parameters. Under this assumption, the loss must be strongly convex and smooth in these parameters; otherwise, the method will diverge for some problems. In this work, we instead assume that the data distribution is Lipschitz continuous with respect to the model's predictions, a more natural assumption for performative systems. As a result, we are able to significantly relax the assumptions on the loss function. In particular, we do not need to assume convexity with respect to the model's parameters. As an illustration, we introduce a resampling procedure that models realisti
    

