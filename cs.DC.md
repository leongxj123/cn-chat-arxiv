# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing.](http://arxiv.org/abs/2308.07541) | 本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。 |
| [^2] | [ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems.](http://arxiv.org/abs/2201.02775) | 本文研究了垂直联邦学习系统中的对抗性主导输入（ADIs），并证明了其在典型VFL系统中的存在。该研究为防止ADIs的使用提供了方法。 |

# 详细

[^1]: 基于强化学习的无服务器计算中冷启动频率降低方法

    Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing. (arXiv:2308.07541v1 [cs.DC])

    [http://arxiv.org/abs/2308.07541](http://arxiv.org/abs/2308.07541)

    本文提出了一种基于强化学习的方法来降低无服务器计算中的冷启动频率。通过使用Q学习和考虑多种指标，我们可以在预期需求的基础上提前初始化函数，从而减少冷启动次数。

    

    函数即服务是一种云计算范例，为应用程序提供了事件驱动执行模型。它通过从开发者那里消除资源管理责任，提供透明和按需可扩展性来实现无服务器特性。典型的无服务器应用程序对响应时间和可扩展性有严格要求，因此依赖于部署的服务为客户提供快速和容错的反馈。然而，函数即服务范例在需要按需初始化函数时存在非常可观的延迟，即冷启动问题。本研究旨在通过使用强化学习来减少平台上的冷启动频率。我们的方法使用Q学习，并考虑函数的CPU利用率、已有函数实例和响应失败率等指标，根据预期需求提前主动初始化函数。我们提出的解决方案在Kubeless上实现并进行评估。

    Function-as-a-Service is a cloud computing paradigm offering an event-driven execution model to applications. It features serverless attributes by eliminating resource management responsibilities from developers and offers transparent and on-demand scalability of applications. Typical serverless applications have stringent response time and scalability requirements and therefore rely on deployed services to provide quick and fault-tolerant feedback to clients. However, the FaaS paradigm suffers from cold starts as there is a non-negligible delay associated with on-demand function initialization. This work focuses on reducing the frequency of cold starts on the platform by using Reinforcement Learning. Our approach uses Q-learning and considers metrics such as function CPU utilization, existing function instances, and response failure rate to proactively initialize functions in advance based on the expected demand. The proposed solution was implemented on Kubeless and was evaluated usin
    
[^2]: ADI: 在垂直联邦学习系统中的对抗性主导输入

    ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems. (arXiv:2201.02775v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2201.02775](http://arxiv.org/abs/2201.02775)

    本文研究了垂直联邦学习系统中的对抗性主导输入（ADIs），并证明了其在典型VFL系统中的存在。该研究为防止ADIs的使用提供了方法。

    

    最近，垂直联邦学习（VFL）系统作为处理分散在许多个体来源中的数据的概念而变得突出，无需将其集中化。多个参与者以隐私意识的方式协作训练基于其本地数据的模型。到目前为止，VFL已成为在组织之间安全学习模型的事实解决方案，允许共享知识而不影响任何个人的隐私。尽管VFL系统的发展昌盛，但我们发现某些参与者的输入，称为对抗性主导输入（ADIs），可以支配共同推断朝着对手的意愿方向并迫使其他（受害者）参与者做出微不足道的贡献，失去通常在联邦学习场景中提供的对其贡献重要性的奖励。我们对ADIs进行了系统研究，首先证明了它们在典型的VFL系统中的存在。然后，我们提出了基于梯度的方法

    Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods
    

