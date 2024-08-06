# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Threats, Attacks, and Defenses in Machine Unlearning: A Survey](https://arxiv.org/abs/2403.13682) | 机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。 |
| [^2] | [InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents](https://arxiv.org/abs/2403.02691) | 本研究引入了InjecAgent基准测试，用于评估工具集成的大型语言模型代理对间接提示注入攻击的脆弱性，通过评估30种LLM代理，发现这些代理存在漏洞 |
| [^3] | [Locally Differentially Private Graph Embedding.](http://arxiv.org/abs/2310.11060) | 该论文提出了一种局部差分隐私图嵌入框架（LDP-GE），该框架采用LDP机制来保护节点数据的隐私，并使用个性化PageRank作为近似度度量来学习节点表示。大量实验证明，LDP-GE在隐私和效用方面取得了有利的折衷效果，并且明显优于现有的方法。 |

# 详细

[^1]: 机器学习中的威胁、攻击和防御：一项调查

    Threats, Attacks, and Defenses in Machine Unlearning: A Survey

    [https://arxiv.org/abs/2403.13682](https://arxiv.org/abs/2403.13682)

    机器遗忘（MU）通过知识去除过程来解决训练数据相关的人工智能治理问题，提高了AI系统的安全和负责任使用。

    

    机器遗忘（MU）最近引起了相当大的关注，因为它有潜力通过从训练的机器学习模型中消除特定数据的影响来实现安全人工智能。这个被称为知识去除的过程解决了与训练数据相关的人工智能治理问题，如数据质量、敏感性、版权限制和过时性。这种能力对于确保遵守诸如被遗忘权等隐私法规也至关重要。此外，有效的知识去除有助于减轻有害结果的风险，防范偏见、误导和未经授权的数据利用，从而增强了AI系统的安全和负责任使用。已经开展了设计高效的遗忘方法的工作，通过研究MU服务以与现有的机器学习作为服务集成，使用户能够提交请求从训练语料库中删除特定数据。

    arXiv:2403.13682v2 Announce Type: replace-cross  Abstract: Machine Unlearning (MU) has gained considerable attention recently for its potential to achieve Safe AI by removing the influence of specific data from trained machine learning models. This process, known as knowledge removal, addresses AI governance concerns of training data such as quality, sensitivity, copyright restrictions, and obsolescence. This capability is also crucial for ensuring compliance with privacy regulations such as the Right To Be Forgotten. Furthermore, effective knowledge removal mitigates the risk of harmful outcomes, safeguarding against biases, misinformation, and unauthorized data exploitation, thereby enhancing the safe and responsible use of AI systems. Efforts have been made to design efficient unlearning approaches, with MU services being examined for integration with existing machine learning as a service, allowing users to submit requests to remove specific data from the training corpus. However, 
    
[^2]: InjecAgent：基于工具集成的大型语言模型Agent中的间接提示注入基准测试

    InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents

    [https://arxiv.org/abs/2403.02691](https://arxiv.org/abs/2403.02691)

    本研究引入了InjecAgent基准测试，用于评估工具集成的大型语言模型代理对间接提示注入攻击的脆弱性，通过评估30种LLM代理，发现这些代理存在漏洞

    

    最近的工作将LLMs作为代理体现出来，使它们能够访问工具，执行操作，并与外部内容（例如，电子邮件或网站）进行交互。然而，外部内容引入了间接提示注入（IPI）攻击的风险，恶意指令被嵌入LLMs处理的内容中，旨在操纵这些代理执行对用户有害的操作。考虑到这类攻击的潜在严重后果，建立用于评估和减轻这些风险的基准测试至关重要。在这项工作中，我们介绍了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent包括1,054个测试用例，涵盖17种不同的用户工具和62种攻击者工具。我们将攻击意图分为两种主要类型：对用户造成直接伤害和窃取私人数据。我们评估了30种不同的LLM代理，并表明这些代理是脆弱的。

    arXiv:2403.02691v1 Announce Type: new  Abstract: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable
    
[^3]: 局部差分隐私图嵌入

    Locally Differentially Private Graph Embedding. (arXiv:2310.11060v1 [cs.CR])

    [http://arxiv.org/abs/2310.11060](http://arxiv.org/abs/2310.11060)

    该论文提出了一种局部差分隐私图嵌入框架（LDP-GE），该框架采用LDP机制来保护节点数据的隐私，并使用个性化PageRank作为近似度度量来学习节点表示。大量实验证明，LDP-GE在隐私和效用方面取得了有利的折衷效果，并且明显优于现有的方法。

    

    图嵌入被证明是学习图中节点潜在表示的强大工具。然而，尽管在各种基于图的机器学习任务中表现出卓越性能，但在涉及敏感信息的图数据上进行学习可能引发重大的隐私问题。为了解决这个问题，本文研究了开发能满足局部差分隐私（LDP）的图嵌入算法的问题。我们提出了一种新颖的隐私保护图嵌入框架LDP-GE，用于保护节点数据的隐私。具体而言，我们提出了一种LDP机制来混淆节点数据，并采用个性化PageRank作为近似度度量来学习节点表示。然后，我们从理论上分析了LDP-GE框架的隐私保证和效用。在几个真实世界的图数据集上进行的大量实验表明，LDP-GE在隐私-效用权衡方面取得了有利的效果，并且明显优于现有的方法。

    Graph embedding has been demonstrated to be a powerful tool for learning latent representations for nodes in a graph. However, despite its superior performance in various graph-based machine learning tasks, learning over graphs can raise significant privacy concerns when graph data involves sensitive information. To address this, in this paper, we investigate the problem of developing graph embedding algorithms that satisfy local differential privacy (LDP). We propose LDP-GE, a novel privacy-preserving graph embedding framework, to protect the privacy of node data. Specifically, we propose an LDP mechanism to obfuscate node data and adopt personalized PageRank as the proximity measure to learn node representations. Then, we theoretically analyze the privacy guarantees and utility of the LDP-GE framework. Extensive experiments conducted over several real-world graph datasets demonstrate that LDP-GE achieves favorable privacy-utility trade-offs and significantly outperforms existing appr
    

