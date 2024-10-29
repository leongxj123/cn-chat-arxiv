# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327) | 该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。 |
| [^2] | [A Hierarchical Framework with Spatio-Temporal Consistency Learning for Emergence Detection in Complex Adaptive Systems.](http://arxiv.org/abs/2401.10300) | 本研究提出了一个分层框架，通过学习系统和代理的表示，使用时空一致性学习来捕捉复杂适应性系统中的现象，并解决了现有方法不能捕捉空间模式和建模非线性关系的问题。 |

# 详细

[^1]: 我们应该交流吗：探索竞争LLM代理之间的自发合作

    Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents

    [https://arxiv.org/abs/2402.12327](https://arxiv.org/abs/2402.12327)

    该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。

    

    最近的进展表明，由大型语言模型（LLMs）驱动的代理具有模拟人类行为和社会动态的能力。然而，尚未研究LLM代理在没有明确指令的情况下自发建立合作关系的潜力。为了弥补这一空白，我们进行了三项案例研究，揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力。这一发现不仅展示了LLM代理模拟人类社会中竞争与合作的能力，也验证了计算社会科学的一个有前途的愿景。具体来说，这表明LLM代理可以用于建模人类社会互动，包括那些自发合作的互动，从而提供对社会现象的洞察。这项研究的源代码可在https://github.com/wuzengqing001225/SABM_ShallWe 找到。

    arXiv:2402.12327v1 Announce Type: new  Abstract: Recent advancements have shown that agents powered by large language models (LLMs) possess capabilities to simulate human behaviors and societal dynamics. However, the potential for LLM agents to spontaneously establish collaborative relationships in the absence of explicit instructions has not been studied. To address this gap, we conduct three case studies, revealing that LLM agents are capable of spontaneously forming collaborations even within competitive settings. This finding not only demonstrates the capacity of LLM agents to mimic competition and cooperation in human societies but also validates a promising vision of computational social science. Specifically, it suggests that LLM agents could be utilized to model human social interactions, including those with spontaneous collaborations, thus offering insights into social phenomena. The source codes for this study are available at https://github.com/wuzengqing001225/SABM_ShallWe
    
[^2]: 用于复杂适应性系统中的现象检测的具有时空一致性学习的分层框架

    A Hierarchical Framework with Spatio-Temporal Consistency Learning for Emergence Detection in Complex Adaptive Systems. (arXiv:2401.10300v1 [cs.MA])

    [http://arxiv.org/abs/2401.10300](http://arxiv.org/abs/2401.10300)

    本研究提出了一个分层框架，通过学习系统和代理的表示，使用时空一致性学习来捕捉复杂适应性系统中的现象，并解决了现有方法不能捕捉空间模式和建模非线性关系的问题。

    

    在由交互代理组成的复杂适应性系统（CAS）中，现象是一种全局属性，在现实世界的动态系统中很普遍，例如网络层次的交通拥堵。检测它的形成和消散有助于监测系统的状态，并发出有害现象的警报信号。由于CAS没有集中式控制器，基于每个代理的局部观察来检测现象是可取但具有挑战性的。现有的工作不能捕捉与现象相关的空间模式，并且无法建模代理之间的非线性关系。本文提出了一个分层框架，通过学习系统表示和代理表示来解决这两个问题，其中时空一致性学习器针对代理的非线性关系和系统的复杂演化进行了定制。通过保留最新100个代理的状态和历史状态来学习代理和系统的表示，

    Emergence, a global property of complex adaptive systems (CASs) constituted by interactive agents, is prevalent in real-world dynamic systems, e.g., network-level traffic congestions. Detecting its formation and evaporation helps to monitor the state of a system, allowing to issue a warning signal for harmful emergent phenomena. Since there is no centralized controller of CAS, detecting emergence based on each agent's local observation is desirable but challenging. Existing works are unable to capture emergence-related spatial patterns, and fail to model the nonlinear relationships among agents. This paper proposes a hierarchical framework with spatio-temporal consistency learning to solve these two problems by learning the system representation and agent representations, respectively. Especially, spatio-temporal encoders are tailored to capture agents' nonlinear relationships and the system's complex evolution. Representations of the agents and the system are learned by preserving the
    

