# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpreting Event-Studies from Recent Difference-in-Differences Methods.](http://arxiv.org/abs/2401.12309) | 本文讨论了最新差异与差异方法产生的事件研究图的解释，发现新方法的图形与传统的两方式固定效应（TWFE）事件研究的图形不匹配，因为新方法将治疗前系数与治疗后系数进行不对称构建。提出了使用这些方法时构建和解释事件研究图的实用建议。 |
| [^2] | [Optimal Conditional Inference in Adaptive Experiments.](http://arxiv.org/abs/2309.12162) | 我们研究了在自适应实验中进行条件推断的问题，证明了在没有进一步限制的情况下，仅使用最后一批结果进行推断是最优的；当实验的自适应方面是位置不变的时，我们还发现了额外的信息；在停止时间、分配概率和目标参数仅依赖于数据的多面体事件集合的情况下，我们推导出了计算可行且最优的条件推断程序。 |

# 详细

[^1]: 从最新的差异与差异方法解读事件研究

    Interpreting Event-Studies from Recent Difference-in-Differences Methods. (arXiv:2401.12309v1 [econ.EM])

    [http://arxiv.org/abs/2401.12309](http://arxiv.org/abs/2401.12309)

    本文讨论了最新差异与差异方法产生的事件研究图的解释，发现新方法的图形与传统的两方式固定效应（TWFE）事件研究的图形不匹配，因为新方法将治疗前系数与治疗后系数进行不对称构建。提出了使用这些方法时构建和解释事件研究图的实用建议。

    

    本文讨论了最新差异与差异方法产生的事件研究图的解释。我展示了即使在非错开治疗时机的情况下专门针对情况进行了特殊化，三种最流行的最新方法（de Chaisemartin和D'Haultfoeuille，2020; Callaway和SantAnna，2021; Borusyak，Jaravel和Spiess，2024）软件生成的默认图形与传统的两方式固定效应（TWFE）事件研究的图形不匹配：新方法可能在治疗时出现转折点或跳跃点，而TWFE事件研究则显示一条直线。这种差异来自于新方法将治疗前系数与治疗后系数进行不对称构建。因此，对于分析TWFE事件研究图的视觉启发，不能立即应用于这些方法的图形上。文章最后给出了在使用这些方法时构建和解释事件研究图的实用建议。

    This note discusses the interpretation of event-study plots produced by recent difference-in-differences methods. I show that even when specialized to the case of non-staggered treatment timing, the default plots produced by software for three of the most popular recent methods (de Chaisemartin and D'Haultfoeuille, 2020; Callaway and SantAnna, 2021; Borusyak, Jaravel and Spiess, 2024) do not match those of traditional two-way fixed effects (TWFE) event-studies: the new methods may show a kink or jump at the time of treatment even when the TWFE event-study shows a straight line. This difference stems from the fact that the new methods construct the pre-treatment coefficients asymmetrically from the post-treatment coefficients. As a result, visual heuristics for analyzing TWFE event-study plots should not be immediately applied to those from these methods. I conclude with practical recommendations for constructing and interpreting event-study plots when using these methods.
    
[^2]: 自适应实验中的最优条件推断

    Optimal Conditional Inference in Adaptive Experiments. (arXiv:2309.12162v1 [stat.ME])

    [http://arxiv.org/abs/2309.12162](http://arxiv.org/abs/2309.12162)

    我们研究了在自适应实验中进行条件推断的问题，证明了在没有进一步限制的情况下，仅使用最后一批结果进行推断是最优的；当实验的自适应方面是位置不变的时，我们还发现了额外的信息；在停止时间、分配概率和目标参数仅依赖于数据的多面体事件集合的情况下，我们推导出了计算可行且最优的条件推断程序。

    

    我们研究了批量赌徒实验，并考虑了在实现停止时间、分配概率和目标参数的条件下进行推断的问题，其中所有这些可能都是根据实验的最后一批信息进行自适应选择的。在没有对实验进行进一步限制的情况下，我们证明仅使用最后一批结果进行推断是最优的。当实验的自适应方面被认为是位置不变的，即当我们将所有批次-臂的平均值都向一个常数移动时，我们证明数据中还存在额外的信息，可以通过一个额外的批次-臂均值的线性函数来捕捉。在更严格的情况下，停止时间、分配概率和目标参数被认为仅依赖于数据通过一个多面体事件的集合，我们推导出了计算可行且最优的条件推断程序。

    We study batched bandit experiments and consider the problem of inference conditional on the realized stopping time, assignment probabilities, and target parameter, where all of these may be chosen adaptively using information up to the last batch of the experiment. Absent further restrictions on the experiment, we show that inference using only the results of the last batch is optimal. When the adaptive aspects of the experiment are known to be location-invariant, in the sense that they are unchanged when we shift all batch-arm means by a constant, we show that there is additional information in the data, captured by one additional linear function of the batch-arm means. In the more restrictive case where the stopping time, assignment probabilities, and target parameter are known to depend on the data only through a collection of polyhedral events, we derive computationally tractable and optimal conditional inference procedures.
    

