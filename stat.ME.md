# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Bayesian Framework for Causal Analysis of Recurrent Events in Presence of Immortal Risk.](http://arxiv.org/abs/2304.03247) | 论文提出了一种贝叶斯框架，针对错位处理问题，将其视为治疗切换问题，并通过概率模型解决了复增和末事件偏差的问题。 |

# 详细

[^1]: 一种在不可避免风险存在下进行复发事件因果分析的贝叶斯框架

    A Bayesian Framework for Causal Analysis of Recurrent Events in Presence of Immortal Risk. (arXiv:2304.03247v1 [stat.ME])

    [http://arxiv.org/abs/2304.03247](http://arxiv.org/abs/2304.03247)

    论文提出了一种贝叶斯框架，针对错位处理问题，将其视为治疗切换问题，并通过概率模型解决了复增和末事件偏差的问题。

    

    生物医学统计学中对复发事件率的观测研究很常见。通常的目标是在规定的随访时间窗口内，估计在一个明确定义的目标人群中两种治疗方法的事件率差异。使用观测性索赔数据进行估计是具有挑战性的，因为在目标人群的成员资格方面定义时，很少在资格确认时准确分配治疗方式。目前的解决方案通常是错位处理，比如基于后续分配，在资格确认时分配治疗方式，这会将先前的事件率错误地归因于治疗-从而产生不可避免的风险偏差。即使资格和治疗已经对齐，终止事件过程（例如死亡）也经常停止感兴趣的复发事件过程。同样，这两个过程也受到审查的影响，因此在整个随访时间窗口内不能观察到事件。我们的方法将错位处理转化为治疗切换问题：一些患者在整个随访时间窗口内坚持一个特定的治疗策略，另一些患者在这个时间窗口内经历治疗策略的切换。我们提出了一个概率模型，其中包括两个基本元素：通过一个合理的时刻切换模型，正确地建模治疗之间的切换和不可避免风险，通过将非观察事件模型化为复发事件模型，解决了复增和末事件偏差的问题。

    Observational studies of recurrent event rates are common in biomedical statistics. Broadly, the goal is to estimate differences in event rates under two treatments within a defined target population over a specified followup window. Estimation with observational claims data is challenging because while membership in the target population is defined in terms of eligibility criteria, treatment is rarely assigned exactly at the time of eligibility. Ad-hoc solutions to this timing misalignment, such as assigning treatment at eligibility based on subsequent assignment, incorrectly attribute prior event rates to treatment - resulting in immortal risk bias. Even if eligibility and treatment are aligned, a terminal event process (e.g. death) often stops the recurrent event process of interest. Both processes are also censored so that events are not observed over the entire followup window. Our approach addresses misalignment by casting it as a treatment switching problem: some patients are on
    

