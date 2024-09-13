# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identifiable causal inference with noisy treatment and no side information.](http://arxiv.org/abs/2306.10614) | 本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。 |
| [^2] | [Identification and multiply robust estimation in causal mediation analysis with treatment noncompliance.](http://arxiv.org/abs/2304.10025) | 本文针对治疗不服从性提出了一种半参数框架来评估因果中介效应，提出了一组假设来识别自然中介效应并推导出成倍稳健估计器。 |
| [^3] | [What is the state of the art? Accounting for multiplicity in machine learning benchmark performance.](http://arxiv.org/abs/2303.07272) | 机器学习基准性能评估中，最先进的（SOTA）性能的估计值过于乐观，容易导致方法的忽视。本文提供了一个概率模型，用于校正多重性偏差并比较方法的性能。 |

# 详细

[^1]: 带有嘈杂治疗和没有侧面信息的可识别因果推断

    Identifiable causal inference with noisy treatment and no side information. (arXiv:2306.10614v1 [cs.LG])

    [http://arxiv.org/abs/2306.10614](http://arxiv.org/abs/2306.10614)

    本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。

    

    在某些因果推断场景中，治疗（即原因）变量的测量存在不准确性，例如在流行病学或计量经济学中。未能纠正测量误差的影响可能导致偏差的因果效应估计。以前的研究没有从因果视角研究解决这个问题的方法，同时允许复杂的非线性依赖关系并且不假设可以访问侧面信息。对于这样的场景，本论文提出了一个模型，它假设存在一个连续的治疗变量，该变量测量不准确。建立在现有测量误差模型的基础上，我们证明了我们的模型的因果效应估计是可识别的，即使没有测量误差方差或其他侧面信息的知识。我们的方法依赖于深度潜在变量模型，其中高斯条件由神经网络参数化，并且我们开发了一个分摊权重变分客观函数来训练该模型。

    In some causal inference scenarios, the treatment (i.e. cause) variable is measured inaccurately, for instance in epidemiology or econometrics. Failure to correct for the effect of this measurement error can lead to biased causal effect estimates. Previous research has not studied methods that address this issue from a causal viewpoint while allowing for complex nonlinear dependencies and without assuming access to side information. For such as scenario, this paper proposes a model that assumes a continuous treatment variable which is inaccurately measured. Building on existing results for measurement error models, we prove that our model's causal effect estimates are identifiable, even without knowledge of the measurement error variance or other side information. Our method relies on a deep latent variable model where Gaussian conditionals are parameterized by neural networks, and we develop an amortized importance-weighted variational objective for training the model. Empirical resul
    
[^2]: 用于因果中介分析中具有治疗不服从性的识别和倍增稳健估计

    Identification and multiply robust estimation in causal mediation analysis with treatment noncompliance. (arXiv:2304.10025v1 [stat.ME])

    [http://arxiv.org/abs/2304.10025](http://arxiv.org/abs/2304.10025)

    本文针对治疗不服从性提出了一种半参数框架来评估因果中介效应，提出了一组假设来识别自然中介效应并推导出成倍稳健估计器。

    

    在实验和观察研究中，人们通常对了解干预方案如何改善最终结果的潜在机制感兴趣。因果中介分析旨在达到此目的，但主要限于治疗完全服从的情况，只有少数情况需要排除限制。在本文中，我们建立了一个半参数框架，用于在无需排除限制的情况下评估具有治疗不服从性的因果中介效应。我们提出了一组假设来识别整个研究人群的自然中介效应，并进一步针对由潜在服从行为特征化的亚人群中的主要自然中介效应进行识别。我们推导出了主要自然中介效应估计量的有效影响函数，这激励了一组倍增稳健估计器进行推论。这些被识别估计量的半参数效率理论。

    In experimental and observational studies, there is often interest in understanding the potential mechanism by which an intervention program improves the final outcome. Causal mediation analyses have been developed for this purpose but are primarily restricted to the case of perfect treatment compliance, with a few exceptions that require exclusion restriction. In this article, we establish a semiparametric framework for assessing causal mediation in the presence of treatment noncompliance without exclusion restriction. We propose a set of assumptions to identify the natural mediation effects for the entire study population and further, for the principal natural mediation effects within subpopulations characterized by the potential compliance behaviour. We derive the efficient influence functions for the principal natural mediation effect estimands, which motivate a set of multiply robust estimators for inference. The semiparametric efficiency theory for the identified estimands is der
    
[^3]: 机器学习基准性能评估中的多重性问题

    What is the state of the art? Accounting for multiplicity in machine learning benchmark performance. (arXiv:2303.07272v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2303.07272](http://arxiv.org/abs/2303.07272)

    机器学习基准性能评估中，最先进的（SOTA）性能的估计值过于乐观，容易导致方法的忽视。本文提供了一个概率模型，用于校正多重性偏差并比较方法的性能。

    

    机器学习方法通常通过在公共数据库中的数据集上的性能来进行评估和比较。这允许多种方法，在相同条件下并跨越时间进行评估。在问题中排名最高的性能被称为最先进的（SOTA）性能，并且被用作新方法出版的参考点。但使用最高排名的性能作为SOTA的估计值是一种有偏的估计器，会给出过于乐观的结果。这种多重性的机制是多重比较和多重检验中广泛研究的主题，但在关于SOTA估计的讨论中几乎没有得到提及。过于乐观的最先进估计值被用作评估新方法的标准，而具有明显劣势结果的方法很容易被忽视。在本文中，我们提供了一个概率模型，用于校正多重性偏差并比较方法的性能。

    Machine learning methods are commonly evaluated and compared by their performance on data sets from public repositories. This allows for multiple methods, oftentimes several thousands, to be evaluated under identical conditions and across time. The highest ranked performance on a problem is referred to as state-of-the-art (SOTA) performance, and is used, among other things, as a reference point for publication of new methods. Using the highest-ranked performance as an estimate for SOTA is a biased estimator, giving overly optimistic results. The mechanisms at play are those of multiplicity, a topic that is well-studied in the context of multiple comparisons and multiple testing, but has, as far as the authors are aware of, been nearly absent from the discussion regarding SOTA estimates. The optimistic state-of-the-art estimate is used as a standard for evaluating new methods, and methods with substantial inferior results are easily overlooked. In this article, we provide a probability 
    

