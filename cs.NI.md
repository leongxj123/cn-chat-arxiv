# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks.](http://arxiv.org/abs/2401.05308) | 该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。 |
| [^2] | [OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models.](http://arxiv.org/abs/2310.07637) | OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。 |

# 详细

[^1]: 面对HAPS使能的FL网络中的非独立同分布问题，战略客户选择的研究

    Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks. (arXiv:2401.05308v1 [cs.NI])

    [http://arxiv.org/abs/2401.05308](http://arxiv.org/abs/2401.05308)

    该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。

    

    在由高空平台站（HAPS）使能的垂直异构网络中部署联合学习（FL）为各种不同通信和计算能力的客户提供了参与的机会。这种多样性不仅提高了FL模型的训练精度，还加快了其收敛速度。然而，在这些广阔的网络中应用FL存在显著的非独立同分布问题。这种数据异质性往往导致收敛速度较慢和模型训练性能的降低。我们的研究引入了一种针对此问题的客户选择策略，利用用户网络流量行为进行预测和分类。该策略通过战略性选择数据呈现相似模式的客户参与，同时优先考虑用户隐私。

    The deployment of federated learning (FL) within vertical heterogeneous networks, such as those enabled by high-altitude platform station (HAPS), offers the opportunity to engage a wide array of clients, each endowed with distinct communication and computational capabilities. This diversity not only enhances the training accuracy of FL models but also hastens their convergence. Yet, applying FL in these expansive networks presents notable challenges, particularly the significant non-IIDness in client data distributions. Such data heterogeneity often results in slower convergence rates and reduced effectiveness in model training performance. Our study introduces a client selection strategy tailored to address this issue, leveraging user network traffic behaviour. This strategy involves the prediction and classification of clients based on their network usage patterns while prioritizing user privacy. By strategically selecting clients whose data exhibit similar patterns for participation
    
[^2]: OpsEval: 用于大型语言模型的全面任务导向的AIOps基准测试

    OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models. (arXiv:2310.07637v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.07637](http://arxiv.org/abs/2310.07637)

    OpsEval是一个全面任务导向的AIOps基准测试，评估了大型语言模型在有线网络操作、5G通信操作和数据库操作等关键场景下的能力水平，为提供针对AIOps定制的LLMs的优化方向。

    

    大型语言模型(Large Language Models, LLMs)在翻译、总结和生成等NLP相关任务中表现出了显著的能力。LLMs在特定领域中应用，特别是在AIOps（面向IT运维的人工智能）中，由于其先进的信息汇总、报告分析和API调用能力而具有巨大的潜力。然而，当前LLMs在AIOps任务中的性能尚未确定。此外，需要一个全面的基准测试来引导针对AIOps定制的LLMs的优化。与现有的专注于评估网络配置等特定领域的基准测试不同，本文提出了OpsEval，这是一个专为LLMs设计的全面任务导向的AIOps基准测试。OpsEval首次对LLMs在三个关键场景（有线网络操作、5G通信操作和数据库操作）以及不同的能力水平（知识回忆、分析思考）进行评估。

    Large language models (LLMs) have exhibited remarkable capabilities in NLP-related tasks such as translation, summarizing, and generation. The application of LLMs in specific areas, notably AIOps (Artificial Intelligence for IT Operations), holds great potential due to their advanced abilities in information summarizing, report analyzing, and ability of API calling. Nevertheless, the performance of current LLMs in AIOps tasks is yet to be determined. Furthermore, a comprehensive benchmark is required to steer the optimization of LLMs tailored for AIOps. Compared with existing benchmarks that focus on evaluating specific fields like network configuration, in this paper, we present \textbf{OpsEval}, a comprehensive task-oriented AIOps benchmark designed for LLMs. For the first time, OpsEval assesses LLMs' proficiency in three crucial scenarios (Wired Network Operation, 5G Communication Operation, and Database Operation) at various ability levels (knowledge recall, analytical thinking, an
    

