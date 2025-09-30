# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity](https://arxiv.org/abs/2403.13374) | 通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。 |
| [^2] | [Decentralized Federated Unlearning on Blockchain](https://arxiv.org/abs/2402.16294) | 提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。 |
| [^3] | [Ocassionally Secure: A Comparative Analysis of Code Generation Assistants](https://arxiv.org/abs/2402.00689) | 本文通过比较分析四种先进的LLMs在9个任务上的表现，确定和理解了在真实场景中有效且安全地部署LLMs生成优质代码的条件和环境。 |

# 详细

[^1]: 具有对数据异构性的自适应的拜占庭弹性联邦学习

    Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity

    [https://arxiv.org/abs/2403.13374](https://arxiv.org/abs/2403.13374)

    通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。

    

    本文处理了在存在恶意拜占庭攻击和数据异构性的情况下的联邦学习（FL）。提出了一种新颖的鲁棒平均梯度算法（RAGA），该算法利用几何中位数进行聚合，并可以自由选择本地更新的轮数。与大多数现有的弹性方法不同，这些方法基于强凸损失函数或均匀分布的数据集进行收敛分析，我们进行了对强凸和非凸损失函数在异构数据集上的收敛分析。根据我们的理论分析，只要恶意用户数据集的比例小于一半，RAGA就可以以$\mathcal{O}({1}/{T^{2/3- \delta}})$的速度实现非凸损失函数的收敛，其中$T$为迭代次数，$\delta \in (0, 2/3)$，对于强凸损失函数则呈线性收敛。此外，稳定点或全局最优解

    arXiv:2403.13374v1 Announce Type: new  Abstract: This paper deals with federated learning (FL) in the presence of malicious Byzantine attacks and data heterogeneity. A novel Robust Average Gradient Algorithm (RAGA) is proposed, which leverages the geometric median for aggregation and can freely select the round number for local updating. Different from most existing resilient approaches, which perform convergence analysis based on strongly-convex loss function or homogeneously distributed dataset, we conduct convergence analysis for not only strongly-convex but also non-convex loss function over heterogeneous dataset. According to our theoretical analysis, as long as the fraction of dataset from malicious users is less than half, RAGA can achieve convergence at rate $\mathcal{O}({1}/{T^{2/3- \delta}})$ where $T$ is the iteration number and $\delta \in (0, 2/3)$ for non-convex loss function, and at linear rate for strongly-convex loss function. Moreover, stationary point or global optim
    
[^2]: 区块链上的去中心化联邦遗忘

    Decentralized Federated Unlearning on Blockchain

    [https://arxiv.org/abs/2402.16294](https://arxiv.org/abs/2402.16294)

    提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。

    

    区块链联邦学习（FL）在确保FL过程的完整性和可追溯性方面越来越受到关注。区块链FL涉及参与者在本地训练模型并随后将模型发布到区块链上，形成表示模型关系的类似有向无环图（DAG）的继承结构。然而，这种基于DAG的结构在使用敏感数据更新模型时存在挑战，因为涉及的复杂性和开销较大。为了解决这个问题，我们提出了基于区块链的联邦遗忘（BlockFUL），这是一个通用框架，使用变色龙哈希（CH）技术重新设计区块链结构，以减轻模型更新的复杂性，从而降低遗忘任务的计算和共识成本。此外，BlockFUL支持各种联邦遗忘方法，确保模型更新的完整性和可追溯性。

    arXiv:2402.16294v1 Announce Type: cross  Abstract: Blockchained Federated Learning (FL) has been gaining traction for ensuring the integrity and traceability of FL processes. Blockchained FL involves participants training models locally with their data and subsequently publishing the models on the blockchain, forming a Directed Acyclic Graph (DAG)-like inheritance structure that represents the model relationship. However, this particular DAG-based structure presents challenges in updating models with sensitive data, due to the complexity and overhead involved. To address this, we propose Blockchained Federated Unlearning (BlockFUL), a generic framework that redesigns the blockchain structure using Chameleon Hash (CH) technology to mitigate the complexity of model updating, thereby reducing the computational and consensus costs of unlearning tasks.Furthermore, BlockFUL supports various federated unlearning methods, ensuring the integrity and traceability of model updates, whether conduc
    
[^3]: 偶尔安全：代码生成辅助工具的比较分析

    Ocassionally Secure: A Comparative Analysis of Code Generation Assistants

    [https://arxiv.org/abs/2402.00689](https://arxiv.org/abs/2402.00689)

    本文通过比较分析四种先进的LLMs在9个任务上的表现，确定和理解了在真实场景中有效且安全地部署LLMs生成优质代码的条件和环境。

    

    大型语言模型(LLMs)在各种应用中的应用越来越广泛，代码生成就是一个显著的例子。以往的研究表明LLMs有能力生成安全和不安全的代码，但文献没有考虑到什么因素有助于生成安全和有效的代码。因此，本文重点是确定和理解在真实场景中LLMs能够有效和安全地部署来生成优质代码的条件和环境。我们对四个先进的LLMs进行了比较分析——使用ChatGPT和Bard的GPT-3.5和GPT-4，以及来自Google的Gemini——使用9个独立任务来评估每个模型的代码生成能力。我们将我们的研究置于一个典型的使用场景中，代表了开发人员在工作中使用LLMs进行日常任务的情况。此外，我们还强调了安全意识，通过使用我们开发的两个不同版本的工具来体现。

    $ $Large Language Models (LLMs) are being increasingly utilized in various applications, with code generations being a notable example. While previous research has shown that LLMs have the capability to generate both secure and insecure code, the literature does not take into account what factors help generate secure and effective code. Therefore in this paper we focus on identifying and understanding the conditions and contexts in which LLMs can be effectively and safely deployed in real-world scenarios to generate quality code. We conducted a comparative analysis of four advanced LLMs--GPT-3.5 and GPT-4 using ChatGPT and Bard and Gemini from Google--using 9 separate tasks to assess each model's code generation capabilities. We contextualized our study to represent the typical use cases of a real-life developer employing LLMs for everyday tasks as work. Additionally, we place an emphasis on security awareness which is represented through the use of two distinct versions of our develop
    

