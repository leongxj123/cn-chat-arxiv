# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Personalized Decision Support Policies.](http://arxiv.org/abs/2304.06701) | 本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。 |

# 详细

[^1]: 学习个性化决策支持策略

    Learning Personalized Decision Support Policies. (arXiv:2304.06701v1 [cs.LG])

    [http://arxiv.org/abs/2304.06701](http://arxiv.org/abs/2304.06701)

    本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    

    个体决策者可能需要不同形式的支持来提高决策结果，但重要的问题是，哪种形式的支持会在低成本下导致准确的决策。本文提出了学习决策支持策略的方法，它在给定输入时选择是否以及如何提供支持。我们考虑没有先验信息的决策者，并将学习各自的策略形式化为一个多目标优化问题，这个问题权衡了准确性和成本。使用随机环境的技术，我们提出了 $\texttt{THREAD}$，这是一种个性化决策支持策略的在线算法，并设计了一种超参数调整策略，以利用模拟人类行为来确定成本-性能权衡。我们提供计算实验来证明 $\texttt{THREAD}$ 相对于线下基线的优势。然后，我们推出了一个交互式工具 $\texttt{Modiste}$，它为现实中的医学诊断提供个性化决策支持。$\texttt{Modiste}$ 使用 $\texttt{THREAD}$ 为每位医生学习个性化的决策支持策略，并推荐个性化研究以优化患者的预期结果并将严重并发症的风险降至最低。使用电子健康记录数据，我们展示了 $\texttt{Modiste}$ 显著提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    Individual human decision-makers may benefit from different forms of support to improve decision outcomes. However, a key question is which form of support will lead to accurate decisions at a low cost. In this work, we propose learning a decision support policy that, for a given input, chooses which form of support, if any, to provide. We consider decision-makers for whom we have no prior information and formalize learning their respective policies as a multi-objective optimization problem that trades off accuracy and cost. Using techniques from stochastic contextual bandits, we propose $\texttt{THREAD}$, an online algorithm to personalize a decision support policy for each decision-maker, and devise a hyper-parameter tuning strategy to identify a cost-performance trade-off using simulated human behavior. We provide computational experiments to demonstrate the benefits of $\texttt{THREAD}$ compared to offline baselines. We then introduce $\texttt{Modiste}$, an interactive tool that pr
    

