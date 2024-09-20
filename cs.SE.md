# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt.](http://arxiv.org/abs/2309.06020) | 本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。 |
| [^2] | [Forecasting the steam mass flow in a powerplant using the parallel hybrid network.](http://arxiv.org/abs/2307.09483) | 这项研究使用并行混合神经网络结构来预测发电厂中的蒸汽质量流量，相比纯经典和纯量子模型，该混合模型在测试集上取得了更好的性能，平均平方误差降低了5.7倍和4.9倍，并且相对误差较小，最多提升了2倍。 |

# 详细

[^1]: 自动评估偿还自认技术债务所需的工作量

    Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt. (arXiv:2309.06020v1 [cs.SE])

    [http://arxiv.org/abs/2309.06020](http://arxiv.org/abs/2309.06020)

    本研究提出了一种新的方法，利用大规模的数据集自动估算自认技术债务的还款工作量。研究结果表明，不同类型的自认技术债务需要不同程度的还款工作量。

    

    技术债务是指在软件开发过程中为了短期利益而做出的次优决策所带来的后果。自认技术债务(SATD)是一种特定形式的技术债务，开发人员明确地在软件的源代码注释和提交消息中记录下来。由于SATD可能阻碍软件的开发和维护，因此有效地解决和优先处理它非常重要。然而，目前的方法缺乏根据SATD的文本描述自动评估其还款工作量的能力。为了解决这个限制，我们提出了一种新的方法，利用一个包括1,060个Apache代码库中共2,568,728个提交的341,740个SATD项目的全面数据集来自动估算SATD还款工作量。我们的研究结果表明，不同类型的SATD需要不同程度的还款工作量，其中代码/设计、需求和测试债务需要更多的工作量。

    Technical debt refers to the consequences of sub-optimal decisions made during software development that prioritize short-term benefits over long-term maintainability. Self-Admitted Technical Debt (SATD) is a specific form of technical debt, explicitly documented by developers within software artifacts such as source code comments and commit messages. As SATD can hinder software development and maintenance, it is crucial to address and prioritize it effectively. However, current methodologies lack the ability to automatically estimate the repayment effort of SATD based on its textual descriptions. To address this limitation, we propose a novel approach for automatically estimating SATD repayment effort, utilizing a comprehensive dataset comprising 341,740 SATD items from 2,568,728 commits across 1,060 Apache repositories. Our findings show that different types of SATD require varying levels of repayment effort, with code/design, requirement, and test debt demanding greater effort compa
    
[^2]: 使用并行混合网络预测发电厂中的蒸汽质量流量

    Forecasting the steam mass flow in a powerplant using the parallel hybrid network. (arXiv:2307.09483v1 [cs.LG])

    [http://arxiv.org/abs/2307.09483](http://arxiv.org/abs/2307.09483)

    这项研究使用并行混合神经网络结构来预测发电厂中的蒸汽质量流量，相比纯经典和纯量子模型，该混合模型在测试集上取得了更好的性能，平均平方误差降低了5.7倍和4.9倍，并且相对误差较小，最多提升了2倍。

    

    高效可持续的发电是能源领域的一个关键问题。尤其是热电厂在准确预测蒸汽质量流量方面面临困难，这对于运营效率和成本降低至关重要。在本研究中，我们使用一个并行混合神经网络结构，该结构将参数化量子电路和传统的前馈神经网络相结合，特别设计用于工业环境中的时间序列预测，以提高对未来15分钟内蒸汽质量流量的预测能力。我们的结果表明，并行混合模型优于独立的经典和量子模型，在训练后的测试集上相对于纯经典模型和纯量子网络，平均平方误差（MSE）损失分别降低了5.7倍和4.9倍。此外，该混合模型在测试集上表现出相对误差较小，比纯经典模型更好，最多提升了2倍。

    Efficient and sustainable power generation is a crucial concern in the energy sector. In particular, thermal power plants grapple with accurately predicting steam mass flow, which is crucial for operational efficiency and cost reduction. In this study, we use a parallel hybrid neural network architecture that combines a parametrized quantum circuit and a conventional feed-forward neural network specifically designed for time-series prediction in industrial settings to enhance predictions of steam mass flow 15 minutes into the future. Our results show that the parallel hybrid model outperforms standalone classical and quantum models, achieving more than 5.7 and 4.9 times lower mean squared error (MSE) loss on the test set after training compared to pure classical and pure quantum networks, respectively. Furthermore, the hybrid model demonstrates smaller relative errors between the ground truth and the model predictions on the test set, up to 2 times better than the pure classical model.
    

