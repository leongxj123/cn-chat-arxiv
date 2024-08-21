# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Response Style Characterization for Repeated Measures Using the Visual Analogue Scale](https://arxiv.org/abs/2403.10136) | 本研究针对重复测量的视觉模拟量表数据开发了一种新颖的响应风格（RP）表征方法，以解决在VAS中处理RP的困难。 |
| [^2] | [Causal Reasoning and Large Language Models: Opening a New Frontier for Causality.](http://arxiv.org/abs/2305.00050) | 大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。 |
| [^3] | [Matrix Quantile Factor Model.](http://arxiv.org/abs/2208.08693) | 本文提出了一种新的矩阵分位因子模型，针对矩阵型数据具有低秩结构。我们通过优化经验核损失函数估计行和列的因子空间，证明了估计值的快速收敛速率，提供了合理的因子数对确定方法，并进行了广泛的模拟研究和实证研究。 |
| [^4] | [Valid Inference after Causal Discovery.](http://arxiv.org/abs/2208.05949) | 本研究开发了工具以实现因果发现后的有效推断，解决了使用相同数据运行因果发现算法后估计因果效应导致经典置信区间的覆盖保证无效问题。 |

# 详细

[^1]: 利用视觉模拟量表对重复测量的响应风格进行表征

    Response Style Characterization for Repeated Measures Using the Visual Analogue Scale

    [https://arxiv.org/abs/2403.10136](https://arxiv.org/abs/2403.10136)

    本研究针对重复测量的视觉模拟量表数据开发了一种新颖的响应风格（RP）表征方法，以解决在VAS中处理RP的困难。

    

    自我报告测量（例如，利克特量表）被广泛用于评估主观健康感知。最近，由于其能够精确且便于评估人们感受的能力，视觉模拟量表（VAS），一种滑动条量表，变得流行起来。这些数据可能会受到响应风格（RS）的影响，RS是一种用户依赖的系统性倾向，无论问卷说明如何都会发生。尽管在个体间分析中尤为重要，但对VAS中RS（表示为响应剖面（RP））的处理并未受到足够关注，因为它主要用于个体内监测且不太受RP的影响。然而，VAS测量通常需要对同一问卷项目进行重复自我报告，这使得难以在利克特量表上应用传统方法。在这项研究中，我们开发了一种新颖的RP表征方法，适用于各种类型的重复测量的VAS数据。

    arXiv:2403.10136v1 Announce Type: cross  Abstract: Self-report measures (e.g., Likert scales) are widely used to evaluate subjective health perceptions. Recently, the visual analog scale (VAS), a slider-based scale, has become popular owing to its ability to precisely and easily assess how people feel. These data can be influenced by the response style (RS), a user-dependent systematic tendency that occurs regardless of questionnaire instructions. Despite its importance, especially in between-individual analysis, little attention has been paid to handling the RS in the VAS (denoted as response profile (RP)), as it is mainly used for within-individual monitoring and is less affected by RP. However, VAS measurements often require repeated self-reports of the same questionnaire items, making it difficult to apply conventional methods on a Likert scale. In this study, we developed a novel RP characterization method for various types of repeatedly measured VAS data. This approach involves t
    
[^2]: 因果推理与大型语言模型：开启因果研究的新篇章

    Causal Reasoning and Large Language Models: Opening a New Frontier for Causality. (arXiv:2305.00050v1 [cs.AI])

    [http://arxiv.org/abs/2305.00050](http://arxiv.org/abs/2305.00050)

    大型语言模型在因果推理任务中取得了新的最高准确率，但是其鲁棒性仍然存在难以预测的失败模式。

    

    大型语言模型的因果能力备受争议，并且对将其应用于医学、科学、法律和政策等具有社会影响力的领域具有重要意义。我们进一步探讨了LLMs及其因果推理的区别，以及潜在的建构和测量效度威胁。基于GPT-3.5和4的算法在多个因果基准测试上取得了新的最高准确率。与此同时，LLMs展示了难以预测的失败模式，我们提供了一些技术来解释它们的鲁棒性。

    The causal capabilities of large language models (LLMs) is a matter of significant debate, with critical implications for the use of LLMs in societally impactful domains such as medicine, science, law, and policy. We further our understanding of LLMs and their causal implications, considering the distinctions between different types of causal reasoning tasks, as well as the entangled threats of construct and measurement validity. LLM-based methods establish new state-of-the-art accuracies on multiple causal benchmarks. Algorithms based on GPT-3.5 and 4 outperform existing algorithms on a pairwise causal discovery task (97%, 13 points gain), counterfactual reasoning task (92%, 20 points gain), and actual causality (86% accuracy in determining necessary and sufficient causes in vignettes). At the same time, LLMs exhibit unpredictable failure modes and we provide some techniques to interpret their robustness.  Crucially, LLMs perform these causal tasks while relying on sources of knowledg
    
[^3]: 矩阵分位因子模型

    Matrix Quantile Factor Model. (arXiv:2208.08693v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.08693](http://arxiv.org/abs/2208.08693)

    本文提出了一种新的矩阵分位因子模型，针对矩阵型数据具有低秩结构。我们通过优化经验核损失函数估计行和列的因子空间，证明了估计值的快速收敛速率，提供了合理的因子数对确定方法，并进行了广泛的模拟研究和实证研究。

    

    本文为具有低秩结构的矩阵型数据引入了矩阵分位因子模型。通过在所有面板上最小化经验核损失函数，我们估计了行和列因子空间。我们证明了这些估计收敛于速率$1/\min\{\sqrt{p_1p_2}, \sqrt{p_2T}, \sqrt{p_1T}\}$在平均Frobenius范数下，其中$p_1$，$p_2$和$T$分别表示矩阵序列的行维数、列维数和长度。该速率比将矩阵模型“展平”为大向量模型的分位估计速率更快。给出了平滑的估计量，并在一些温和的条件下导出了它们的中心极限定理。我们提供了三个一致的标准来确定行和列因子数对。广泛的模拟研究和实证研究验证了我们的理论。

    This paper introduces a matrix quantile factor model for matrix-valued data with a low-rank structure. We estimate the row and column factor spaces via minimizing the empirical check loss function over all panels. We show the estimates converge at rate $1/\min\{\sqrt{p_1p_2}, \sqrt{p_2T},$ $\sqrt{p_1T}\}$ in average Frobenius norm, where $p_1$, $p_2$ and $T$ are the row dimensionality, column dimensionality and length of the matrix sequence. This rate is faster than that of the quantile estimates via ``flattening" the matrix model into a large vector model. Smoothed estimates are given and their central limit theorems are derived under some mild condition. We provide three consistent criteria to determine the pair of row and column factor numbers. Extensive simulation studies and an empirical study justify our theory.
    
[^4]: 因果发现后的有效推断

    Valid Inference after Causal Discovery. (arXiv:2208.05949v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.05949](http://arxiv.org/abs/2208.05949)

    本研究开发了工具以实现因果发现后的有效推断，解决了使用相同数据运行因果发现算法后估计因果效应导致经典置信区间的覆盖保证无效问题。

    

    因果发现和因果效应估计是因果推断中的两个基本任务。虽然已经针对每个任务单独开发了许多方法，但是同时应用这些方法时会出现统计上的挑战：在对相同数据运行因果发现算法后估计因果效应会导致"双重挑选"，从而使经典置信区间的覆盖保证无效。为此，我们开发了针对因果发现后有效的推断工具。通过实证研究，我们发现，天真组合因果发现算法和随后推断算法会导致高度膨胀的误覆盖率，而应用我们的方法则提供可靠的覆盖并实现比数据分割更准确的因果发现。

    Causal discovery and causal effect estimation are two fundamental tasks in causal inference. While many methods have been developed for each task individually, statistical challenges arise when applying these methods jointly: estimating causal effects after running causal discovery algorithms on the same data leads to "double dipping," invalidating the coverage guarantees of classical confidence intervals. To this end, we develop tools for valid post-causal-discovery inference. Across empirical studies, we show that a naive combination of causal discovery and subsequent inference algorithms leads to highly inflated miscoverage rates; on the other hand, applying our method provides reliable coverage while achieving more accurate causal discovery than data splitting.
    

