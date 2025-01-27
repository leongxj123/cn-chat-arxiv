# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What's in a Name? Auditing Large Language Models for Race and Gender Bias](https://arxiv.org/abs/2402.14875) | 调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。 |
| [^2] | [Learning Personalized Decision Support Policies.](http://arxiv.org/abs/2304.06701) | 本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。 |

# 详细

[^1]: 名字的含义是什么？审计大型语言模型中的种族和性别偏见

    What's in a Name? Auditing Large Language Models for Race and Gender Bias

    [https://arxiv.org/abs/2402.14875](https://arxiv.org/abs/2402.14875)

    调查发现，大型语言模型存在种族和性别偏见，尤其对与黑人女性相关的名字表现最不利。审计在模型部署和实施时的重要性得到强调。

    

    我们采用审计设计来调查最先进的大型语言模型中的偏见，包括GPT-4。在我们的研究中，我们引发模型在各种情景下为个人提供建议，比如在购车谈判或选举结果预测过程中。我们发现该建议系统性地对与种族少数群体和女性常见相关的名字产生不利影响。与黑人女性相关的名字得到的结果最不利。这些偏见在42个提示模板和多个模型中都是一致的，表明这是一个系统性问题，而不是孤立事件。在提示中提供数值、与决策相关的锚点可以成功抵消偏见，而定性细节的影响并不一致，甚至可能会加剧差异。我们的研究结果强调了在语言模型部署和实施时进行审计的重要性，以减轻其潜在影响。

    arXiv:2402.14875v1 Announce Type: cross  Abstract: We employ an audit design to investigate biases in state-of-the-art large language models, including GPT-4. In our study, we elicit prompt the models for advice regarding an individual across a variety of scenarios, such as during car purchase negotiations or election outcome predictions. We find that the advice systematically disadvantages names that are commonly associated with racial minorities and women. Names associated with Black women receive the least advantageous outcomes. The biases are consistent across 42 prompt templates and several models, indicating a systemic issue rather than isolated incidents. While providing numerical, decision-relevant anchors in the prompt can successfully counteract the biases, qualitative details have inconsistent effects and may even increase disparities. Our findings underscore the importance of conducting audits at the point of LLM deployment and implementation to mitigate their potential for
    
[^2]: 学习个性化决策支持策略

    Learning Personalized Decision Support Policies. (arXiv:2304.06701v1 [cs.LG])

    [http://arxiv.org/abs/2304.06701](http://arxiv.org/abs/2304.06701)

    本文提出了一种学习个性化决策支持策略的算法 $\texttt{THREAD}$，可以为决策者提供不同形式的支持。同时，引入了 $\texttt{Modiste}$ 工具来提供个性化的医学诊断决策支持，使用 $\texttt{THREAD}$ 学习个性化决策支持策略，有效提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    

    个体决策者可能需要不同形式的支持来提高决策结果，但重要的问题是，哪种形式的支持会在低成本下导致准确的决策。本文提出了学习决策支持策略的方法，它在给定输入时选择是否以及如何提供支持。我们考虑没有先验信息的决策者，并将学习各自的策略形式化为一个多目标优化问题，这个问题权衡了准确性和成本。使用随机环境的技术，我们提出了 $\texttt{THREAD}$，这是一种个性化决策支持策略的在线算法，并设计了一种超参数调整策略，以利用模拟人类行为来确定成本-性能权衡。我们提供计算实验来证明 $\texttt{THREAD}$ 相对于线下基线的优势。然后，我们推出了一个交互式工具 $\texttt{Modiste}$，它为现实中的医学诊断提供个性化决策支持。$\texttt{Modiste}$ 使用 $\texttt{THREAD}$ 为每位医生学习个性化的决策支持策略，并推荐个性化研究以优化患者的预期结果并将严重并发症的风险降至最低。使用电子健康记录数据，我们展示了 $\texttt{Modiste}$ 显著提高了预期的诊断正确性，并减少了严重并发症的风险，同时推荐了更少和更便宜的研究。

    Individual human decision-makers may benefit from different forms of support to improve decision outcomes. However, a key question is which form of support will lead to accurate decisions at a low cost. In this work, we propose learning a decision support policy that, for a given input, chooses which form of support, if any, to provide. We consider decision-makers for whom we have no prior information and formalize learning their respective policies as a multi-objective optimization problem that trades off accuracy and cost. Using techniques from stochastic contextual bandits, we propose $\texttt{THREAD}$, an online algorithm to personalize a decision support policy for each decision-maker, and devise a hyper-parameter tuning strategy to identify a cost-performance trade-off using simulated human behavior. We provide computational experiments to demonstrate the benefits of $\texttt{THREAD}$ compared to offline baselines. We then introduce $\texttt{Modiste}$, an interactive tool that pr
    

