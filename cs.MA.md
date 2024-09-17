# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal Off-Policy Prediction for Multi-Agent Systems](https://arxiv.org/abs/2403.16871) | 这项工作介绍了MA-COPP，这是第一个解决涉及多智能体系统的离策略预测问题的一致预测方法。 |
| [^2] | [A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2403.10996) | 提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题 |
| [^3] | [S-Agents: self-organizing agents in open-ended environment](https://arxiv.org/abs/2402.04578) | S-Agents是一个自组织代理系统，通过引入代理树结构、沙漏代理架构和非阻塞协作方法，实现了在无限环境中高效协调代理的能力，提供了优化协作效率和灵活性的解决方案。 |

# 详细

[^1]: 多智能体系统的一致离策略预测

    Conformal Off-Policy Prediction for Multi-Agent Systems

    [https://arxiv.org/abs/2403.16871](https://arxiv.org/abs/2403.16871)

    这项工作介绍了MA-COPP，这是第一个解决涉及多智能体系统的离策略预测问题的一致预测方法。

    

    离策略预测（OPP），即仅使用在一个正常（行为）策略下收集的数据来预测目标策略的结果，在数据驱动的安全关键系统分析中是一个重要问题，在这种系统中，部署新策略可能是不安全的。为了实现可信的离策略预测，最近关于一致离策略预测（COPP）的工作利用一致预测框架来在目标过程下推导带有概率保证的预测区域。现有的COPP方法可以考虑由策略切换引起的分布偏移，但仅限于单智能体系统和标量结果（例如，奖励）。在这项工作中，我们介绍了MA-COPP，这是第一个解决涉及多智能体系统的OPP问题的一致预测方法，在一个或多个“自我”智能体改变策略时为所有智能体轨迹推导联合预测区域。与单智能体场景不同，这种情况下

    arXiv:2403.16871v1 Announce Type: cross  Abstract: Off-Policy Prediction (OPP), i.e., predicting the outcomes of a target policy using only data collected under a nominal (behavioural) policy, is a paramount problem in data-driven analysis of safety-critical systems where the deployment of a new policy may be unsafe. To achieve dependable off-policy predictions, recent work on Conformal Off-Policy Prediction (COPP) leverage the conformal prediction framework to derive prediction regions with probabilistic guarantees under the target process. Existing COPP methods can account for the distribution shifts induced by policy switching, but are limited to single-agent systems and scalar outcomes (e.g., rewards). In this work, we introduce MA-COPP, the first conformal prediction method to solve OPP problems involving multi-agent systems, deriving joint prediction regions for all agents' trajectories when one or more "ego" agents change their policies. Unlike the single-agent scenario, this se
    
[^2]: 一个可扩展且可并行化的数字孪生框架，用于多智能体强化学习系统可持续Sim2Real转换

    A Scalable and Parallelizable Digital Twin Framework for Sustainable Sim2Real Transition of Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2403.10996](https://arxiv.org/abs/2403.10996)

    提出了一个可持续的多智能体深度强化学习框架，利用分散的学习架构，来解决交通路口穿越和自主赛车等问题

    

    本工作提出了一个可持续的多智能体深度强化学习框架，能够选择性地按需扩展并行化训练工作负载，并利用最少的硬件资源将训练好的策略从模拟环境转移到现实世界。我们引入了AutoDRIVE生态系统作为一个启动数字孪生框架，用于训练、部署和转移合作和竞争的多智能体强化学习策略从模拟环境到现实世界。具体来说，我们首先探究了4台合作车辆(Nigel)在单智能体和多智能体学习环境中共享有限状态信息的交叉遍历问题，采用了一种通用策略方法。然后，我们使用个体策略方法研究了2辆车(F1TENTH)的对抗性自主赛车问题。在任何一组实验中，我们采用了去中心化学习架构，这允许对策略进行有力的训练和测试。

    arXiv:2403.10996v1 Announce Type: cross  Abstract: This work presents a sustainable multi-agent deep reinforcement learning framework capable of selectively scaling parallelized training workloads on-demand, and transferring the trained policies from simulation to reality using minimal hardware resources. We introduce AutoDRIVE Ecosystem as an enabling digital twin framework to train, deploy, and transfer cooperative as well as competitive multi-agent reinforcement learning policies from simulation to reality. Particularly, we first investigate an intersection traversal problem of 4 cooperative vehicles (Nigel) that share limited state information in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial autonomous racing problem of 2 vehicles (F1TENTH) using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the policies 
    
[^3]: S-Agents: 自组织代理在无限环境中的应用

    S-Agents: self-organizing agents in open-ended environment

    [https://arxiv.org/abs/2402.04578](https://arxiv.org/abs/2402.04578)

    S-Agents是一个自组织代理系统，通过引入代理树结构、沙漏代理架构和非阻塞协作方法，实现了在无限环境中高效协调代理的能力，提供了优化协作效率和灵活性的解决方案。

    

    利用大型语言模型（LLMs），自主代理能够显著提升，具备处理各种任务的能力。在无限环境中，为了提高效率和效果，优化协作需要灵活的调整。然而，当前的研究主要强调固定的、任务导向的工作流程，忽视了以代理为中心的组织结构。受人类组织行为的启发，我们引入了一种自组织代理系统（S-Agents），其中包括动态工作流程的“代理树”结构、平衡信息优先级的“沙漏代理架构”以及允许代理之间异步执行任务的“非阻塞协作”方法。这种结构可以自主协调一组代理，有效地解决无限且动态的环境挑战，无需人工干预。我们的实验表明，S-Agents能够熟练地执行协作建筑任务和资源收集。

    Leveraging large language models (LLMs), autonomous agents have significantly improved, gaining the ability to handle a variety of tasks. In open-ended settings, optimizing collaboration for efficiency and effectiveness demands flexible adjustments. Despite this, current research mainly emphasizes fixed, task-oriented workflows and overlooks agent-centric organizational structures. Drawing inspiration from human organizational behavior, we introduce a self-organizing agent system (S-Agents) with a "tree of agents" structure for dynamic workflow, an "hourglass agent architecture" for balancing information priorities, and a "non-obstructive collaboration" method to allow asynchronous task execution among agents. This structure can autonomously coordinate a group of agents, efficiently addressing the challenges of an open and dynamic environment without human intervention. Our experiments demonstrate that S-Agents proficiently execute collaborative building tasks and resource collection i
    

