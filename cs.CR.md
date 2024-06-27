# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Privacy-Preserving Model Explanations: Privacy Risks, Attacks, and Countermeasures](https://arxiv.org/abs/2404.00673) | 本研究是第一个全面调查模型解释中隐私攻击及其对抗措施的论文，通过分类隐私攻击和对抗措施，初步探讨了隐私泄漏原因，提出未解决问题和未来研究方向。 |
| [^2] | [The Fundamental Limits of Least-Privilege Learning](https://arxiv.org/abs/2402.12235) | 最小权限学习存在一个基本的权衡，即表示对于给定任务的实用性和其泄漏到任务外属性之间存在无法避免的权衡。 |
| [^3] | [SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2402.03741) | 该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。 |

# 详细

[^1]: 隐私保护型模型解释研究综述：隐私风险、攻击和对抗措施

    A Survey of Privacy-Preserving Model Explanations: Privacy Risks, Attacks, and Countermeasures

    [https://arxiv.org/abs/2404.00673](https://arxiv.org/abs/2404.00673)

    本研究是第一个全面调查模型解释中隐私攻击及其对抗措施的论文，通过分类隐私攻击和对抗措施，初步探讨了隐私泄漏原因，提出未解决问题和未来研究方向。

    

    随着可解释人工智能（XAI）的采用不断扩大，解决其隐私影响的紧迫性变得更加迫切。尽管在人工智能隐私和可解释性方面有越来越多的研究，但对于隐私保护型模型解释却鲜有关注。本文首次全面调查了模型解释的隐私攻击及其对抗措施。我们在这一领域的贡献包括对研究论文进行彻底分析，并提供了一个相互连接的分类法，便于根据目标解释对隐私攻击和对抗措施进行分类。本研究还对隐私泄漏原因进行了初步调查。最后，我们讨论了我们分析中发现的未解决问题和未来研究方向。该调查旨在成为研究界的宝贵资源，并为这一领域的新手提供明确的见解。

    arXiv:2404.00673v1 Announce Type: cross  Abstract: As the adoption of explainable AI (XAI) continues to expand, the urgency to address its privacy implications intensifies. Despite a growing corpus of research in AI privacy and explainability, there is little attention on privacy-preserving model explanations. This article presents the first thorough survey about privacy attacks on model explanations and their countermeasures. Our contribution to this field comprises a thorough analysis of research papers with a connected taxonomy that facilitates the categorisation of privacy attacks and countermeasures based on the targeted explanations. This work also includes an initial investigation into the causes of privacy leaks. Finally, we discuss unresolved issues and prospective research directions uncovered in our analysis. This survey aims to be a valuable resource for the research community and offers clear insights for those new to this domain. To support ongoing research, we have estab
    
[^2]: 最小权限学习的基本限制

    The Fundamental Limits of Least-Privilege Learning

    [https://arxiv.org/abs/2402.12235](https://arxiv.org/abs/2402.12235)

    最小权限学习存在一个基本的权衡，即表示对于给定任务的实用性和其泄漏到任务外属性之间存在无法避免的权衡。

    

    最少权限学习的承诺是找到对于学习任务有用的特征表示，但同时防止推断与该任务无关的任何敏感信息，这一点非常吸引人。然而，到目前为止，这个概念只是以非正式的方式陈述。因此，我们仍然不清楚我们是否以及如何实现这个目标。在这项工作中，我们首次为机器学习的最小权限原则提供了形式化，并描述了其可行性。我们证明了在表示对于给定任务的实用性和其泄漏到预期任务之外的属性之间存在基本权衡：不可能学习到对于预期任务具有高实用性的表示，同时又防止推断除任务标签本身之外的任何属性。这种权衡是无论使用何种技术来学习产生这些表示的特征映射都是成立的。我们经验性地验证了这一点。

    arXiv:2402.12235v1 Announce Type: new  Abstract: The promise of least-privilege learning -- to find feature representations that are useful for a learning task but prevent inference of any sensitive information unrelated to this task -- is highly appealing. However, so far this concept has only been stated informally. It thus remains an open question whether and how we can achieve this goal. In this work, we provide the first formalisation of the least-privilege principle for machine learning and characterise its feasibility. We prove that there is a fundamental trade-off between a representation's utility for a given task and its leakage beyond the intended task: it is not possible to learn representations that have high utility for the intended task but, at the same time prevent inference of any attribute other than the task label itself. This trade-off holds regardless of the technique used to learn the feature mappings that produce these representations. We empirically validate thi
    
[^3]: SUB-PLAY：针对部分观测的多智能体强化学习系统的对抗策略

    SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2402.03741](https://arxiv.org/abs/2402.03741)

    该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。

    

    最近在多智能体强化学习（MARL）领域取得的进展为无人机的群体控制、机械臂的协作操纵以及多目标包围等开辟了广阔的应用前景。然而，在MARL部署过程中存在潜在的安全威胁需要更多关注和深入调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞生成对抗策略，导致受害者在特定任务中失败。例如，将超人级别的围棋AI的获胜率降低到约20%。这些研究主要关注两人竞争环境，并假设攻击者具有完整的全局状态观测。

    Recent advances in multi-agent reinforcement learning (MARL) have opened up vast application prospects, including swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent researches reveal that an attacker can rapidly exploit the victim's vulnerabilities and generate adversarial policies, leading to the victim's failure in specific tasks. For example, reducing the winning rate of a superhuman-level Go AI to around 20%. They predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY), which incorporate
    

