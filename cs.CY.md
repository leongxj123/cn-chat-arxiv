# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SecGPT: An Execution Isolation Architecture for LLM-Based Systems](https://arxiv.org/abs/2403.04960) | 提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题 |

# 详细

[^1]: SecGPT：一种面向基于LLM系统的执行隔离架构

    SecGPT: An Execution Isolation Architecture for LLM-Based Systems

    [https://arxiv.org/abs/2403.04960](https://arxiv.org/abs/2403.04960)

    提出了一种面向LLM系统的执行隔离架构SecGPT，旨在解决第三方应用程序执行所引发的安全和隐私问题

    

    大型语言模型（LLMs）被扩展为系统，如ChatGPT，已经开始支持第三方应用程序。这些LLM应用程序利用LLMs的事实上基于自然语言的自动执行范式：即，应用程序及其交互是用自然语言定义的，提供对用户数据的访问，并被允许自由地相互交互以及与系统互动。这些LLM应用程序生态系统类似于早期计算平台的设置，在那里应用程序和系统之间缺乏足够的隔离。由于第三方应用程序可能不可信，并且受自然语言界面的不精确性加剧，当前的设计会为用户带来安全和隐私风险。在本文中，我们提出了SecGPT，一种面向LLM系统的架构，旨在缓解由第三方应用程序执行引起的安全性和隐私问题。SecGPT的关键思想是隔离应用程序的执行和更多的预

    arXiv:2403.04960v1 Announce Type: cross  Abstract: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more pre
    

