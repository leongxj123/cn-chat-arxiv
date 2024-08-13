# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution](https://rss.arxiv.org/abs/2402.01586) | 本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。 |
| [^2] | [Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing](https://arxiv.org/abs/2402.02097) | 提出了一种名为MACE的简单而有效的多智能体协同探索方法，通过共享局部新颖性来近似全局新颖性，并引入加权互信息来衡量智能体行动对其他智能体的影响，从而促进多智能体之间的协同探索。实验证明，MACE在稀疏奖励的多智能体环境中取得了出色的性能。 |

# 详细

[^1]: TrustAgent: 通过代理构成实现安全可信赖的LLM代理

    TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution

    [https://rss.arxiv.org/abs/2402.01586](https://rss.arxiv.org/abs/2402.01586)

    本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。

    

    近年来，基于LLM的代理引起了广泛关注，但其可信度仍未得到深入探索。由于代理可以直接与物理环境交互，其可靠性和安全性至关重要。本文提出了一种基于代理构成的代理框架TrustAgent，对LLM代理的安全性维度进行了初步研究。该框架包括三种策略：预先规划策略，在生成计划之前向模型注入安全知识；规划过程中策略，在生成计划时增强安全性；计划后检查策略，通过计划后检查确保安全性。通过实验分析，我们展示了这些方法如何通过识别和预防潜在危险有效提高LLM代理的安全性。此外，我们还探讨了安全性与使用者满意度之间的复杂关系，以及模型的推理能力与其效率之间的关联。

    The emergence of LLM-based agents has garnered considerable attention, yet their trustworthiness remains an under-explored area. As agents can directly interact with the physical environment, their reliability and safety is critical. This paper presents an Agent-Constitution-based agent framework, TrustAgent, an initial investigation into improving the safety dimension of trustworthiness in LLM-based agents. This framework consists of threefold strategies: pre-planning strategy which injects safety knowledge to the model prior to plan generation, in-planning strategy which bolsters safety during plan generation, and post-planning strategy which ensures safety by post-planning inspection. Through experimental analysis, we demonstrate how these approaches can effectively elevate an LLM agent's safety by identifying and preventing potential dangers. Furthermore, we explore the intricate relationships between safety and helpfulness, and between the model's reasoning ability and its efficac
    
[^2]: 利用新颖性共享解决分散式多智能体协同探索问题

    Settling Decentralized Multi-Agent Coordinated Exploration by Novelty Sharing

    [https://arxiv.org/abs/2402.02097](https://arxiv.org/abs/2402.02097)

    提出了一种名为MACE的简单而有效的多智能体协同探索方法，通过共享局部新颖性来近似全局新颖性，并引入加权互信息来衡量智能体行动对其他智能体的影响，从而促进多智能体之间的协同探索。实验证明，MACE在稀疏奖励的多智能体环境中取得了出色的性能。

    

    分散式协作式多智能体强化学习中的探索面临两个挑战。一是全局状态的新颖性不可用，而局部观察的新颖性存在偏差。另一个挑战是智能体如何协调地进行探索。为了解决这些挑战，我们提出了一种简单但有效的多智能体协同探索方法MACE。通过仅传播局部新颖性，智能体可以考虑其他智能体的局部新颖性来近似全局新颖性。此外，我们新引入了加权互信息来衡量一个智能体的行动对其他智能体累计新颖性的影响。我们将其作为内在回报来鼓励智能体对其他智能体的探索产生更大的影响，从而促进协同探索。实验证明，MACE在三种稀疏奖励的多智能体环境中表现出优异的性能。

    Exploration in decentralized cooperative multi-agent reinforcement learning faces two challenges. One is that the novelty of global states is unavailable, while the novelty of local observations is biased. The other is how agents can explore in a coordinated way. To address these challenges, we propose MACE, a simple yet effective multi-agent coordinated exploration method. By communicating only local novelty, agents can take into account other agents' local novelty to approximate the global novelty. Further, we newly introduce weighted mutual information to measure the influence of one agent's action on other agents' accumulated novelty. We convert it as an intrinsic reward in hindsight to encourage agents to exert more influence on other agents' exploration and boost coordinated exploration. Empirically, we show that MACE achieves superior performance in three multi-agent environments with sparse rewards.
    

