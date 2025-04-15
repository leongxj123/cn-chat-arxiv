# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^2] | [Evaluating AI systems under uncertain ground truth: a case study in dermatology.](http://arxiv.org/abs/2307.02191) | 这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。 |

# 详细

[^1]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^2]: 在不确定的基准事实下评估AI系统：皮肤病例研究

    Evaluating AI systems under uncertain ground truth: a case study in dermatology. (arXiv:2307.02191v1 [cs.LG])

    [http://arxiv.org/abs/2307.02191](http://arxiv.org/abs/2307.02191)

    这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。

    

    为了安全起见，在部署之前，卫生领域的AI系统需要经过全面的评估，将其预测结果与假定为确定的基准事实进行验证。然而，实际情况并非如此，基准事实可能是不确定的。不幸的是，在标准的AI模型评估中，这一点被大部分忽视了，但是它可能会产生严重后果，如高估未来的性能。为了避免这种情况，我们测量了基准事实的不确定性，我们假设它可以分解为两个主要部分：注释不确定性是由于缺乏可靠注释，以及由于有限的观测信息而导致的固有不确定性。在确定地聚合注释时，通常会忽视这种基准事实的不确定性，例如通过多数投票或平均值来聚合。相反，我们提出了一个框架，在该框架中使用统计模型进行注释的聚合。具体而言，我们将注释的聚合框架解释为所谓可能性的后验推断。

    For safety, AI systems in health undergo thorough evaluations before deployment, validating their predictions against a ground truth that is assumed certain. However, this is actually not the case and the ground truth may be uncertain. Unfortunately, this is largely ignored in standard evaluation of AI models but can have severe consequences such as overestimating the future performance. To avoid this, we measure the effects of ground truth uncertainty, which we assume decomposes into two main components: annotation uncertainty which stems from the lack of reliable annotations, and inherent uncertainty due to limited observational information. This ground truth uncertainty is ignored when estimating the ground truth by deterministically aggregating annotations, e.g., by majority voting or averaging. In contrast, we propose a framework where aggregation is done using a statistical model. Specifically, we frame aggregation of annotations as posterior inference of so-called plausibilities
    

