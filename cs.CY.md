# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterpart Fairness -- Addressing Systematic between-group Differences in Fairness Evaluation.](http://arxiv.org/abs/2305.18160) | 本论文提出了一种新的公平评估方法，通过比较不同群体中相似的个体来解决群体之间的系统差异。这种方法基于倾向得分，识别对等个体，避免了比较不同类型的个体，从而提高公平性评估的准确性和可靠性。 |

# 详细

[^1]: 对等公平性——解决公平评估中群体之间系统差异的问题

    Counterpart Fairness -- Addressing Systematic between-group Differences in Fairness Evaluation. (arXiv:2305.18160v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.18160](http://arxiv.org/abs/2305.18160)

    本论文提出了一种新的公平评估方法，通过比较不同群体中相似的个体来解决群体之间的系统差异。这种方法基于倾向得分，识别对等个体，避免了比较不同类型的个体，从而提高公平性评估的准确性和可靠性。

    

    当使用机器学习（ML）辅助决策时，确保算法决策公平性至关重要，即不歧视特定个体/群体，尤其是来自弱势群体的人。现有的群体公平方法要求进行平等的群体测量，但未考虑群体之间的系统差异。混淆因素，这些因素虽然与敏感变量无关，但表现出系统差异，会对公平评估产生重要影响。为解决这个问题，我们认为公平测量应该基于不同群体中相似于感兴趣任务的对等人（即彼此相似的个体）之间的比较，其群体身份不可通过探索混淆因素算法地区分。我们开发了基于倾向得分的方法来识别对等个体，以避免公平评估比较“橙子”和“苹果”。

    When using machine learning (ML) to aid decision-making, it is critical to ensure that an algorithmic decision is fair, i.e., it does not discriminate against specific individuals/groups, particularly those from underprivileged populations. Existing group fairness methods require equal group-wise measures, which however fails to consider systematic between-group differences. The confounding factors, which are non-sensitive variables but manifest systematic differences, can significantly affect fairness evaluation. To tackle this problem, we believe that a fairness measurement should be based on the comparison between counterparts (i.e., individuals who are similar to each other with respect to the task of interest) from different groups, whose group identities cannot be distinguished algorithmically by exploring confounding factors. We have developed a propensity-score-based method for identifying counterparts, which prevents fairness evaluation from comparing "oranges" with "apples". 
    

