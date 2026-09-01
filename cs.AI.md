# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^2] | [SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2402.03741) | 该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。 |
| [^3] | [General Phrase Debiaser: Debiasing Masked Language Models at a Multi-Token Level.](http://arxiv.org/abs/2311.13892) | 本文提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够有效减轻掩码语言模型中的短语级别偏见，并在标准数据集和指标上取得了最新成果。 |
| [^4] | [Potential Energy Advantage of Quantum Economy.](http://arxiv.org/abs/2308.08025) | 量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。 |

# 详细

[^1]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^2]: SUB-PLAY：针对部分观测的多智能体强化学习系统的对抗策略

    SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2402.03741](https://arxiv.org/abs/2402.03741)

    该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。

    

    最近在多智能体强化学习（MARL）领域取得的进展为无人机的群体控制、机械臂的协作操纵以及多目标包围等开辟了广阔的应用前景。然而，在MARL部署过程中存在潜在的安全威胁需要更多关注和深入调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞生成对抗策略，导致受害者在特定任务中失败。例如，将超人级别的围棋AI的获胜率降低到约20%。这些研究主要关注两人竞争环境，并假设攻击者具有完整的全局状态观测。

    Recent advances in multi-agent reinforcement learning (MARL) have opened up vast application prospects, including swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent researches reveal that an attacker can rapidly exploit the victim's vulnerabilities and generate adversarial policies, leading to the victim's failure in specific tasks. For example, reducing the winning rate of a superhuman-level Go AI to around 20%. They predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY), which incorporate
    
[^3]: 通用短语去偏器：在多标记级别上消除掩码语言模型中的偏见

    General Phrase Debiaser: Debiasing Masked Language Models at a Multi-Token Level. (arXiv:2311.13892v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.13892](http://arxiv.org/abs/2311.13892)

    本文提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够有效减轻掩码语言模型中的短语级别偏见，并在标准数据集和指标上取得了最新成果。

    

    预训练语言模型所揭示的社会偏见和不受欢迎的刻板印象正在成为其应用的障碍。与针对词级别的众多去偏方法相比，对于短语级别的偏见关注相对较少，限制了学科领域去偏的性能。在本文中，我们提出了一种名为“通用短语去偏器”的自动多标记去偏管道，能够减轻掩码语言模型中的短语级别偏见。具体而言，我们的方法包括一个“短语过滤阶段”，从维基百科页面中生成刻板印象的短语，以及一个“模型去偏阶段”，可以在多标记级别上去偏模型以应对短语上的偏见挑战。后者寻找触发模型偏见的提示，然后将其用于去偏。标准数据集和评估指标上的最新成果表明，我们的方法可以显著减少职业和加强性别偏见。

    The social biases and unwelcome stereotypes revealed by pretrained language models are becoming obstacles to their application. Compared to numerous debiasing methods targeting word level, there has been relatively less attention on biases present at phrase level, limiting the performance of debiasing in discipline domains. In this paper, we propose an automatic multi-token debiasing pipeline called \textbf{General Phrase Debiaser}, which is capable of mitigating phrase-level biases in masked language models. Specifically, our method consists of a \textit{phrase filter stage} that generates stereotypical phrases from Wikipedia pages as well as a \textit{model debias stage} that can debias models at the multi-token level to tackle bias challenges on phrases. The latter searches for prompts that trigger model's bias, and then uses them for debiasing. State-of-the-art results on standard datasets and metrics show that our approach can significantly reduce gender biases on both career and 
    
[^4]: 量子经济的势能优势

    Potential Energy Advantage of Quantum Economy. (arXiv:2308.08025v1 [quant-ph])

    [http://arxiv.org/abs/2308.08025](http://arxiv.org/abs/2308.08025)

    量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。

    

    随着大规模机器学习模型和语言模型的广泛部署，能源成本越来越关键。对于提供计算服务的公司来说，低能耗对于市场增长和政府法规来说都非常重要。本文研究了量子计算与经典计算之间的能源优势。我们在能源效率的背景下重新定义优势，与仅基于计算复杂性的传统量子优势不同。通过一个以能量使用为约束条件的Cournot竞争模型，我们证明量子计算公司在Nash均衡点上在盈利能力和能源效率方面都能超越经典对手。因此，量子计算可能代表计算行业更可持续的发展路径。此外，我们发现量子计算经济的能源利益取决于大规模计算。

    Energy cost is increasingly crucial in the modern computing industry with the wide deployment of large-scale machine learning models and language models. For the firms that provide computing services, low energy consumption is important both from the perspective of their own market growth and the government's regulations. In this paper, we study the energy benefits of quantum computing vis-a-vis classical computing. Deviating from the conventional notion of quantum advantage based solely on computational complexity, we redefine advantage in an energy efficiency context. Through a Cournot competition model constrained by energy usage, we demonstrate quantum computing firms can outperform classical counterparts in both profitability and energy efficiency at Nash equilibrium. Therefore quantum computing may represent a more sustainable pathway for the computing industry. Moreover, we discover that the energy benefits of quantum computing economies are contingent on large-scale computation
    

