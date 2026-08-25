# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |
| [^2] | [Revealed Preferences of One-Sided Matching.](http://arxiv.org/abs/2210.14388) | 本文研究了单边匹配的显露偏好，提出了合理化的概念，并在非可转让和可转让效用设置中进行了研究。 |
| [^3] | [Detecting Grouped Local Average Treatment Effects and Selecting True Instruments.](http://arxiv.org/abs/2207.04481) | 我们提出了一个两步骤的过程来识别具有相同局部平均处理效应的响应组，并且即使有一些工具违反了识别假设，也可以选择真正满足假设的工具。 |

# 详细

[^1]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    
[^2]: 单边匹配的显露偏好

    Revealed Preferences of One-Sided Matching. (arXiv:2210.14388v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2210.14388](http://arxiv.org/abs/2210.14388)

    本文研究了单边匹配的显露偏好，提出了合理化的概念，并在非可转让和可转让效用设置中进行了研究。

    

    考虑Shapley和Scarf（1974）的对象分配（单边匹配）模型。当观察到最终分配但代理的偏好未知时，什么情况下分配可能在核心中？这是Echenique，Lee，Shum和Yenmez（2013）模型的单边模拟。我建立了一个模型，在这个模型中，严格的核心是可以测试的，即如果存在一个偏好配置把它置于核心中，则分配是“合理化”的。通过这种方式，我发展了一种单边匹配的显露偏好理论。我在非可转让和可转让效用设置中研究了合理性。在非可转让效用设置中，只有当具有相同偏好的代理处于相同的潜在交易周期中时，他们才会获得相同的分配，这时分配是合理化的。在可转让效用设置中，只有当存在一个价格向量支持分配作为竞争均衡时，分配才是合理化的。

    Consider the object allocation (one-sided matching) model of Shapley and Scarf (1974). When final allocations are observed but agents' preferences are unknown, when might the allocation be in the core? This is a one-sided analogue of the model in Echenique, Lee, Shum, and Yenmez (2013). I build a model in which the strict core is testable -- an allocation is "rationalizable" if there is a preference profile putting it in the core. In this manner, I develop a theory of the revealed preferences of one-sided matching. I study rationalizability in both non-transferrable and transferrable utility settings. In the non-transferrable utility setting, an allocation is rationalizable if and only if: whenever agents with the same preferences are in the same potential trading cycle, they receive the same allocation. In the transferrable utility setting, an allocation is rationalizable if and only if: there exists a price vector supporting the allocation as a competitive equilibrium; or equivalentl
    
[^3]: 检测分组的局部平均处理效应并选择真正的工具

    Detecting Grouped Local Average Treatment Effects and Selecting True Instruments. (arXiv:2207.04481v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2207.04481](http://arxiv.org/abs/2207.04481)

    我们提出了一个两步骤的过程来识别具有相同局部平均处理效应的响应组，并且即使有一些工具违反了识别假设，也可以选择真正满足假设的工具。

    

    在具有异质效应和多个工具的内生二元处理中，我们提出了一个两步骤的过程，用于识别具有相同局部平均处理效应（LATE）的响应组，尽管依赖于不同的工具，即使有几个工具违反了识别假设。我们利用了LATE对于满足LATE假设（工具有效性和处理在工具上单调性）和在给定相应工具的情况下生成相同响应组的工具来说是均匀的这一事实。我们提出了一个两步骤的过程，第一步我们首先聚类倾向得分，在第二步中找到具有相同减少形式参数的IV组。在众数假设下，对于具有相同处理倾向的工具集合，真正满足LATE假设的工具是最大的组，我们的方法可以识别出这些真正的工具。

    Under an endogenous binary treatment with heterogeneous effects and multiple instruments, we propose a two-step procedure for identifying complier groups with identical local average treatment effects (LATE) despite relying on distinct instruments, even if several instruments violate the identifying assumptions. We use the fact that the LATE is homogeneous for instruments which (i) satisfy the LATE assumptions (instrument validity and treatment monotonicity in the instrument) and (ii) generate identical complier groups in terms of treatment propensities given the respective instruments. We propose a two-step procedure, where we first cluster the propensity scores in the first step and find groups of IVs with the same reduced form parameters in the second step. Under the plurality assumption that within each set of instruments with identical treatment propensities, instruments truly satisfying the LATE assumptions are the largest group, our procedure permits identifying these true instr
    

