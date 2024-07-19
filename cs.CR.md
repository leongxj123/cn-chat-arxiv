# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data](https://arxiv.org/abs/2403.10663) | 通过使用多视角数据为深度神经网络添加水印，可以有效防御对源模型功能的窃取攻击 |
| [^2] | [Can LLMs Patch Security Issues?.](http://arxiv.org/abs/2312.00024) | 本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。 |
| [^3] | [Improved Membership Inference Attacks Against Language Classification Models.](http://arxiv.org/abs/2310.07219) | 在这篇论文中，我们提出了一个新的框架，用于对语言分类模型进行成员推理攻击。通过利用集成方法，生成多个专门的攻击模型，我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型更准确。 |
| [^4] | [Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features.](http://arxiv.org/abs/2308.16391) | 这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。 |
| [^5] | [Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters.](http://arxiv.org/abs/2306.14088) | 本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。 |

# 详细

[^1]: 不仅改变标签，学习特征：使用多视角数据为深度神经网络添加水印

    Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data

    [https://arxiv.org/abs/2403.10663](https://arxiv.org/abs/2403.10663)

    通过使用多视角数据为深度神经网络添加水印，可以有效防御对源模型功能的窃取攻击

    

    随着机器学习作为服务（MLaaS）平台的日益普及，越来越多关注深度神经网络（DNN）水印技术。这些方法用于验证目标DNN模型的所有权以保护知识产权。本文首先从特征学习的角度引入了一种新颖的基于触发集的水印方法。具体来说，我们表明通过选择展示多个特征的数据，也被称为$\textit{多视角数据}$，可以有效地防御...

    arXiv:2403.10663v1 Announce Type: cross  Abstract: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as $\textit{multi-view data}$, it becomes feasible to effectively defend 
    
[^2]: LLMs能够修复安全问题吗？

    Can LLMs Patch Security Issues?. (arXiv:2312.00024v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.00024](http://arxiv.org/abs/2312.00024)

    本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。

    

    大型语言模型(LLMs)在代码生成方面显示出了令人印象深刻的能力。然而，类似于人类开发者，这些模型可能会生成包含安全漏洞和缺陷的代码。编写安全代码仍然是一个重大挑战，因为漏洞通常在程序与外部系统或服务（如数据库和操作系统）之间的交互过程中出现。在本文中，我们提出了一种新颖的方法，即基于反馈的解决方案合成（FDSS），旨在探索使用LLMs接收来自静态代码分析工具Bandit的反馈，然后LLMs生成潜在解决方案来解决安全漏洞。每个解决方案以及易受攻击的代码随后被送回LLMs进行代码完善。我们的方法在基线上表现出显著改进，并优于现有方法。此外，我们引入了一个新的数据集PythonSecurityEval，该数据集收集了来自Stack Overflow的真实场景数据。

    Large Language Models (LLMs) have shown impressive proficiency in code generation. Nonetheless, similar to human developers, these models might generate code that contains security vulnerabilities and flaws. Writing secure code remains a substantial challenge, as vulnerabilities often arise during interactions between programs and external systems or services, such as databases and operating systems. In this paper, we propose a novel approach, Feedback-Driven Solution Synthesis (FDSS), designed to explore the use of LLMs in receiving feedback from Bandit, which is a static code analysis tool, and then the LLMs generate potential solutions to resolve security vulnerabilities. Each solution, along with the vulnerable code, is then sent back to the LLM for code refinement. Our approach shows a significant improvement over the baseline and outperforms existing approaches. Furthermore, we introduce a new dataset, PythonSecurityEval, collected from real-world scenarios on Stack Overflow to e
    
[^3]: 改进的对语言分类模型的成员推理攻击

    Improved Membership Inference Attacks Against Language Classification Models. (arXiv:2310.07219v1 [cs.LG])

    [http://arxiv.org/abs/2310.07219](http://arxiv.org/abs/2310.07219)

    在这篇论文中，我们提出了一个新的框架，用于对语言分类模型进行成员推理攻击。通过利用集成方法，生成多个专门的攻击模型，我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型更准确。

    

    人工智能系统在日常生活中普遍存在，具有零售、制造、健康等许多领域的用例。随着人工智能采用的增加，已经发现了相关的风险，包括对使用其数据训练模型的人的隐私风险。评估机器学习模型的隐私风险对于是否使用、部署或共享模型做出知情决策至关重要。隐私风险评估的一种常见方法是对模型进行一个或多个已知攻击，并测量它们的成功率。我们提出了一个新颖的框架，用于对分类模型进行成员推理攻击。我们的框架利用集成方法，为不同数据子集生成许多专门的攻击模型。我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型都实现了更高的准确性。

    Artificial intelligence systems are prevalent in everyday life, with use cases in retail, manufacturing, health, and many other fields. With the rise in AI adoption, associated risks have been identified, including privacy risks to the people whose data was used to train models. Assessing the privacy risks of machine learning models is crucial to enabling knowledgeable decisions on whether to use, deploy, or share a model. A common approach to privacy risk assessment is to run one or more known attacks against the model and measure their success rate. We present a novel framework for running membership inference attacks against classification models. Our framework takes advantage of the ensemble method, generating many specialized attack models for different subsets of the data. We show that this approach achieves higher accuracy than either a single attack model or an attack model per class label, both on classical and language classification tasks.
    
[^4]: 提高以太坊上庞氏骗局检测的鲁棒性和准确性的方法

    Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features. (arXiv:2308.16391v1 [cs.CR])

    [http://arxiv.org/abs/2308.16391](http://arxiv.org/abs/2308.16391)

    这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。

    

    区块链的快速发展导致越来越多的资金涌入加密货币市场，也吸引了近年来网络犯罪分子的兴趣。庞氏骗局作为一种老式的欺诈行为，现在也流行于区块链上，给许多加密货币投资者造成了巨大的财务损失。现有文献中已经提出了一些庞氏骗局检测方法，其中大多数是基于智能合约的源代码或操作码进行检测的。虽然基于合约代码的方法在准确性方面表现出色，但它缺乏鲁棒性：首先，大部分以太坊上的合约源代码并不公开可用；其次，庞氏骗局开发者可以通过混淆操作码或者创造新的分配逻辑来欺骗基于合约代码的检测模型（因为这些模型仅在现有的庞氏逻辑上进行训练）。基于交易的方法可以提高检测的鲁棒性，因为与智能合约不同，交易更加难以伪装。

    The rapid development of blockchain has led to more and more funding pouring into the cryptocurrency market, which also attracted cybercriminals' interest in recent years. The Ponzi scheme, an old-fashioned fraud, is now popular on the blockchain, causing considerable financial losses to many crypto-investors. A few Ponzi detection methods have been proposed in the literature, most of which detect a Ponzi scheme based on its smart contract source code or opcode. The contract-code-based approach, while achieving very high accuracy, is not robust: first, the source codes of a majority of contracts on Ethereum are not available, and second, a Ponzi developer can fool a contract-code-based detection model by obfuscating the opcode or inventing a new profit distribution logic that cannot be detected (since these models were trained on existing Ponzi logics only). A transaction-based approach could improve the robustness of detection because transactions, unlike smart contracts, are harder t
    
[^5]: 非同质化集群下的无线联邦学习中的私有数据聚合

    Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters. (arXiv:2306.14088v1 [cs.LG])

    [http://arxiv.org/abs/2306.14088](http://arxiv.org/abs/2306.14088)

    本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。

    

    联邦学习是通过多个参与客户端私有数据的协同训练神经网络的方法。在训练神经网络的过程中，使用一种著名并广泛使用的迭代优化算法——梯度下降算法。每个客户端使用本地数据计算局部梯度并将其发送给联合器以进行聚合。客户端数据的隐私是一个主要问题。实际上，观察到局部梯度就足以泄露客户端的数据。已研究了用于应对联邦学习中隐私问题的私有聚合方案，其中所有用户都彼此连接并与联合器连接。本文考虑了一个无线系统架构，其中客户端仅通过基站连接到联合器。当需要信息论隐私时，我们推导出通信成本的基本极限，并引入和分析了一种针对这种情况量身定制的私有聚合方案。

    Federated learning collaboratively trains a neural network on privately owned data held by several participating clients. The gradient descent algorithm, a well-known and popular iterative optimization procedure, is run to train the neural network. Every client uses its local data to compute partial gradients and sends it to the federator which aggregates the results. Privacy of the clients' data is a major concern. In fact, observing the partial gradients can be enough to reveal the clients' data. Private aggregation schemes have been investigated to tackle the privacy problem in federated learning where all the users are connected to each other and to the federator. In this paper, we consider a wireless system architecture where clients are only connected to the federator via base stations. We derive fundamental limits on the communication cost when information-theoretic privacy is required, and introduce and analyze a private aggregation scheme tailored for this setting.
    

