# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Information Elicitation in Agency Games](https://arxiv.org/abs/2402.14005) | 该论文探讨了在代理游戏中如何通过代理的信息揭示来指导度量标准的选择，以解决信息不足的问题。 |
| [^2] | [Pure Monte Carlo Counterfactual Regret Minimization.](http://arxiv.org/abs/2309.03084) | 纯蒙特卡洛反事实遗憾最小化算法（PCFR）是一种结合了反事实遗憾最小化（CFR）和虚拟游戏（FP）概念的新算法，能够与各种CFR变体相结合，包括蒙特卡洛CFR（MCCFR）。PCFR具有更好的性能和较快的收敛速度，同时降低了时间和空间复杂度。 |

# 详细

[^1]: 代理游戏中的信息引导

    Information Elicitation in Agency Games

    [https://arxiv.org/abs/2402.14005](https://arxiv.org/abs/2402.14005)

    该论文探讨了在代理游戏中如何通过代理的信息揭示来指导度量标准的选择，以解决信息不足的问题。

    

    在数据收集和数据处理方面的可扩展化、通用化工具取得了快速进展，这使得公司和政策制定者能够采用更加复杂的度量标准作为决策指南成为可能。然而，决定计算哪些度量标准仍然是一个挑战，尤其是对于那些尚未知晓的信息限制下，公司的广泛计算度量标准的能力并不能解决“未知的未知”问题。为了在面对这种信息问题时指导度量标准的选择，我们转向评估代理本身，他们可能比委托人拥有更多关于如何有效衡量结果的信息。我们将这种互动建模为一个简单的代理游戏，询问：代理何时有动机向委托人揭示与成本相关变量的可观察性？存在两种效应：更好的信息降低了代理商的不透明度

    arXiv:2402.14005v1 Announce Type: cross  Abstract: Rapid progress in scalable, commoditized tools for data collection and data processing has made it possible for firms and policymakers to employ ever more complex metrics as guides for decision-making. These developments have highlighted a prevailing challenge -- deciding *which* metrics to compute. In particular, a firm's ability to compute a wider range of existing metrics does not address the problem of *unknown unknowns*, which reflects informational limitations on the part of the firm. To guide the choice of metrics in the face of this informational problem, we turn to the evaluated agents themselves, who may have more information than a principal about how to measure outcomes effectively. We model this interaction as a simple agency game, where we ask: *When does an agent have an incentive to reveal the observability of a cost-correlated variable to the principal?* There are two effects: better information reduces the agent's inf
    
[^2]: 纯蒙特卡洛反事实遗憾最小化

    Pure Monte Carlo Counterfactual Regret Minimization. (arXiv:2309.03084v1 [cs.AI])

    [http://arxiv.org/abs/2309.03084](http://arxiv.org/abs/2309.03084)

    纯蒙特卡洛反事实遗憾最小化算法（PCFR）是一种结合了反事实遗憾最小化（CFR）和虚拟游戏（FP）概念的新算法，能够与各种CFR变体相结合，包括蒙特卡洛CFR（MCCFR）。PCFR具有更好的性能和较快的收敛速度，同时降低了时间和空间复杂度。

    

    反事实遗憾最小化（CFR）及其变体是目前解决大规模不完全信息博弈的最佳算法。本文在CFR的基础上提出了一种名为纯CFR（PCFR）的新算法，以实现更好的性能。PCFR可以看作是CFR和虚拟游戏（FP）的结合，继承了CFR的反事实遗憾（值）的概念，并在下一次迭代中使用最佳响应策略而不是遗憾匹配策略。我们的理论证明了PCFR可以实现Blackwell可达性，使PCFR能够与包括蒙特卡洛CFR（MCCFR）在内的任何CFR变体相结合。由此产生的纯MCCFR（PMCCFR）可以大大降低时间和空间复杂度。特别地，PMCCFR的收敛速度至少比MCCFR快三倍。此外，由于PMCCFR不通过严格被支配策略的路径，我们开发了一种新的启动算法，受到了严格被支配策略的启示。

    Counterfactual Regret Minimization (CFR) and its variants are the best algorithms so far for solving large-scale incomplete information games. Building upon CFR, this paper proposes a new algorithm named Pure CFR (PCFR) for achieving better performance. PCFR can be seen as a combination of CFR and Fictitious Play (FP), inheriting the concept of counterfactual regret (value) from CFR, and using the best response strategy instead of the regret matching strategy for the next iteration. Our theoretical proof that PCFR can achieve Blackwell approachability enables PCFR's ability to combine with any CFR variant including Monte Carlo CFR (MCCFR). The resultant Pure MCCFR (PMCCFR) can significantly reduce time and space complexity. Particularly, the convergence speed of PMCCFR is at least three times more than that of MCCFR. In addition, since PMCCFR does not pass through the path of strictly dominated strategies, we developed a new warm-start algorithm inspired by the strictly dominated strat
    

