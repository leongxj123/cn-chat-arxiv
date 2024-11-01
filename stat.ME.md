# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Natural Counterfactuals With Necessary Backtracking](https://rss.arxiv.org/abs/2402.01607) | 本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。 |
| [^2] | [Quasi-randomization tests for network interference](https://arxiv.org/abs/2403.16673) | 构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。 |
| [^3] | [Self-Consistent Conformal Prediction](https://arxiv.org/abs/2402.07307) | 自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。 |
| [^4] | [Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms](https://arxiv.org/abs/2402.04952) | 本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。 |
| [^5] | [Discovery of the Hidden World with Large Language Models](https://arxiv.org/abs/2402.03941) | 通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。 |
| [^6] | [A powerful rank-based correction to multiple testing under positive dependency.](http://arxiv.org/abs/2311.10900) | 我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。 |
| [^7] | [Controlling Continuous Relaxation for Combinatorial Optimization.](http://arxiv.org/abs/2309.16965) | 本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。 |

# 详细

[^1]: 具有必要回溯的自然反事实

    Natural Counterfactuals With Necessary Backtracking

    [https://rss.arxiv.org/abs/2402.01607](https://rss.arxiv.org/abs/2402.01607)

    本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。

    

    反事实推理对于人类认知非常重要，尤其对于提供解释和做出决策至关重要。尽管Judea Pearl的研究方法在理论上很优雅，但其生成反事实情景往往需要过于脱离实际情景的干预，因此难以实施。为了解决这个问题，我们提出了一种自然反事实的框架和一种根据实际世界数据分布生成自然反事实的方法。我们的方法提供了对反事实推理的改进，允许对因果前置变量进行改变以最小化与实际情景的偏差。为了生成自然反事实，我们引入了一种创新的优化框架，通过自然性准则允许但控制回溯的范围。实证实验表明了我们方法的有效性。

    Counterfactual reasoning is pivotal in human cognition and especially important for providing explanations and making decisions. While Judea Pearl's influential approach is theoretically elegant, its generation of a counterfactual scenario often requires interventions that are too detached from the real scenarios to be feasible. In response, we propose a framework of natural counterfactuals and a method for generating counterfactuals that are natural with respect to the actual world's data distribution. Our methodology refines counterfactual reasoning, allowing changes in causally preceding variables to minimize deviations from realistic scenarios. To generate natural counterfactuals, we introduce an innovative optimization framework that permits but controls the extent of backtracking with a naturalness criterion. Empirical experiments indicate the effectiveness of our method.
    
[^2]: 网络干扰的准随机化检验

    Quasi-randomization tests for network interference

    [https://arxiv.org/abs/2403.16673](https://arxiv.org/abs/2403.16673)

    构建条件准随机化检验来解决网络中干扰存在时的推理问题，使零假设在受限人口上成为尖锐。

    

    许多经典的推理方法在人口单位之间存在干扰时失效。这意味着一个单位的处理状态会影响人口中其他单位的潜在结果。在这种情况下测试这种影响的零假设会使零假设非尖锐。解决这种设置中零假设非尖锐性的一个有趣方法是构建条件随机化检验，使得零假设在受限人口上是尖锐的。在随机实验中，条件随机化检验具有有限样本有效性。这种方法可能会带来计算挑战，因为根据实验设计找到这些适当的子人口可能涉及解决一个NP难的问题。在这篇论文中，我们将人口之间的网络视为一个随机变量而不是固定的。我们提出了一种建立条件准随机化检验的新方法。我们的主要思想是

    arXiv:2403.16673v1 Announce Type: cross  Abstract: Many classical inferential approaches fail to hold when interference exists among the population units. This amounts to the treatment status of one unit affecting the potential outcome of other units in the population. Testing for such spillover effects in this setting makes the null hypothesis non-sharp. An interesting approach to tackling the non-sharp nature of the null hypothesis in this setup is constructing conditional randomization tests such that the null is sharp on the restricted population. In randomized experiments, conditional randomized tests hold finite sample validity. Such approaches can pose computational challenges as finding these appropriate sub-populations based on experimental design can involve solving an NP-hard problem. In this paper, we view the network amongst the population as a random variable instead of being fixed. We propose a new approach that builds a conditional quasi-randomization test. Our main ide
    
[^3]: 自洽的符合预测

    Self-Consistent Conformal Prediction

    [https://arxiv.org/abs/2402.07307](https://arxiv.org/abs/2402.07307)

    自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。

    

    在机器学习指导下的决策中，决策者通常在具有相同预测结果的情境中采取相同的行动。符合预测帮助决策者量化动作的结果不确定性，从而实现更好的风险管理。受这种观点的启发，我们引入了自洽的符合预测，它产生了既符合Venn-Abers校准的预测，又符合以模型预测引发的动作为条件的符合预测区间。我们的方法可以后验地应用于任何黑盒预测器，提供严格的、针对具体动作的决策保证。数值实验表明，我们的方法在区间的效率和条件的有效性之间达到了平衡。

    In decision-making guided by machine learning, decision-makers often take identical actions in contexts with identical predicted outcomes. Conformal prediction helps decision-makers quantify outcome uncertainty for actions, allowing for better risk management. Inspired by this perspective, we introduce self-consistent conformal prediction, which yields both Venn-Abers calibrated predictions and conformal prediction intervals that are valid conditional on actions prompted by model predictions. Our procedure can be applied post-hoc to any black-box predictor to provide rigorous, action-specific decision-making guarantees. Numerical experiments show our approach strikes a balance between interval efficiency and conditional validity.
    
[^4]: 评估因果推断算法的马尔科夫等价类指标

    Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms

    [https://arxiv.org/abs/2402.04952](https://arxiv.org/abs/2402.04952)

    本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。

    

    许多最先进的因果推断方法旨在生成一个输出图，该图编码了生成数据过程的因果图的图形分离和连接陈述。在本文中，我们认为，对合成数据的因果推断方法进行评估应该包括分析该方法的输出与真实情况的分离/连接程度，以衡量这一明确目标的实现情况。我们证明现有的评估指标不能准确捕捉到两个因果图的分离/连接差异，并引入了三个新的距离度量指标，即s/c距离、马尔科夫距离和忠实度距离，以解决这个问题。我们通过玩具示例、实证实验和伪代码来补充我们的理论分析。

    Many state-of-the-art causal discovery methods aim to generate an output graph that encodes the graphical separation and connection statements of the causal graph that underlies the data-generating process. In this work, we argue that an evaluation of a causal discovery method against synthetic data should include an analysis of how well this explicit goal is achieved by measuring how closely the separations/connections of the method's output align with those of the ground truth. We show that established evaluation measures do not accurately capture the difference in separations/connections of two causal graphs, and we introduce three new measures of distance called s/c-distance, Markov distance and Faithfulness distance that address this shortcoming. We complement our theoretical analysis with toy examples, empirical experiments and pseudocode.
    
[^5]: 用大型语言模型探索隐藏世界

    Discovery of the Hidden World with Large Language Models

    [https://arxiv.org/abs/2402.03941](https://arxiv.org/abs/2402.03941)

    通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。

    

    科学起源于从已知事实和观察中发现新的因果知识。传统的因果发现方法主要依赖于高质量的测量变量，通常由人类专家提供，以找到因果关系。然而，在许多现实世界的应用中，因果变量通常无法获取。大型语言模型（LLMs）的崛起为从原始观测数据中发现高级隐藏变量提供了新的机会。因此，我们介绍了COAT：因果表示助手。COAT将LLMs作为因素提供器引入，提取出来自非结构化数据的潜在因果因子。此外，LLMs还可以被指示提供用于收集数据值（例如注释标准）的额外信息，并将原始非结构化数据进一步解析为结构化数据。注释数据将被输入到...

    Science originates with discovering new causal knowledge from a combination of known facts and observations. Traditional causal discovery approaches mainly rely on high-quality measured variables, usually given by human experts, to find causal relations. However, the causal variables are usually unavailable in a wide range of real-world applications. The rise of large language models (LLMs) that are trained to learn rich knowledge from the massive observations of the world, provides a new opportunity to assist with discovering high-level hidden variables from the raw observational data. Therefore, we introduce COAT: Causal representatiOn AssistanT. COAT incorporates LLMs as a factor proposer that extracts the potential causal factors from unstructured data. Moreover, LLMs can also be instructed to provide additional information used to collect data values (e.g., annotation criteria) and to further parse the raw unstructured data into structured data. The annotated data will be fed to a
    
[^6]: 一种基于秩的多重检验正相关依赖的强大修正方法

    A powerful rank-based correction to multiple testing under positive dependency. (arXiv:2311.10900v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.10900](http://arxiv.org/abs/2311.10900)

    我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。

    

    我们开发了一种能够高效利用可能相关的统计假设检验之间正相关性的家族误差率(FWER)控制的新型多重假设检验修正算法$\texttt{max-rank}$。我们的方法概念上很直观，依赖于在计算的统计检验的秩域使用$\max$算子。通过理论和经验的比较，我们证明了在存在正相关依赖的情况下，我们的方法优于经常使用的Bonferroni修正，而在不存在正相关依赖的情况下等效。我们的优势随着测试数量的增加而增加，同时在保证FWER控制的情况下保持高统计功效。我们特别将我们的算法应用于并行置换检验的背景中，这是在我们主要应用的一种复杂预测场景中产生的情况下。

    We develop a novel multiple hypothesis testing correction with family-wise error rate (FWER) control that efficiently exploits positive dependencies between potentially correlated statistical hypothesis tests. Our proposed algorithm $\texttt{max-rank}$ is conceptually straight-forward, relying on the use of a $\max$-operator in the rank domain of computed test statistics. We compare our approach to the frequently employed Bonferroni correction, theoretically and empirically demonstrating its superiority over Bonferroni in the case of existing positive dependency, and its equivalence otherwise. Our advantage over Bonferroni increases as the number of tests rises, and we maintain high statistical power whilst ensuring FWER control. We specifically frame our algorithm in the context of parallel permutation testing, a scenario that arises in our primary application of conformal prediction, a recently popularized approach for quantifying uncertainty in complex predictive settings.
    
[^7]: 控制组合优化的连续放松

    Controlling Continuous Relaxation for Combinatorial Optimization. (arXiv:2309.16965v1 [stat.ML])

    [http://arxiv.org/abs/2309.16965](http://arxiv.org/abs/2309.16965)

    本文研究了在相对密集的图上组合优化问题中物理启发的图神经网络（PI-GNN）求解器的表现。通过数值实验，我们发现PI-GNN求解器在学习早期可能陷入所有变量为零的局部解。为了解决这个问题，我们通过控制连续性和离散性提出了一种改进方法。

    

    最近在组合优化（CO）问题中，图神经网络（GNNs）显示出巨大潜力。通过无监督学习找到近似解的受物理启发的GNN（PI-GNN）求解器在大规模CO问题上引起了极大关注。然而，对于相对密集图上的CO问题，贪婪算法的性能恶化，但对于PI-GNN求解器的性能却没有太多讨论。此外，由于PI-GNN求解器采用了放松策略，学习后需要从连续空间人工转换回原始离散空间，可能会破坏解的鲁棒性。本文通过数值实验证明了PI-GNN求解器在密集图上的CO问题的学习早期可能陷入局部解的情况，其中所有变量都为零。然后，我们通过控制连续性和离散性来解决这些问题。

    Recent advancements in combinatorial optimization (CO) problems emphasize the potential of graph neural networks (GNNs). The physics-inspired GNN (PI-GNN) solver, which finds approximate solutions through unsupervised learning, has attracted significant attention for large-scale CO problems. Nevertheless, there has been limited discussion on the performance of the PI-GNN solver for CO problems on relatively dense graphs where the performance of greedy algorithms worsens. In addition, since the PI-GNN solver employs a relaxation strategy, an artificial transformation from the continuous space back to the original discrete space is necessary after learning, potentially undermining the robustness of the solutions. This paper numerically demonstrates that the PI-GNN solver can be trapped in a local solution, where all variables are zero, in the early stage of learning for CO problems on the dense graphs. Then, we address these problems by controlling the continuity and discreteness of rela
    

