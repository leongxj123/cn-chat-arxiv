# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Comprehensive Study of the Capabilities of Large Language Models for Vulnerability Detection](https://arxiv.org/abs/2403.17218) | 本研究调查了十一种领先的大型语言模型在漏洞检测中的能力，并评估了它们的性能，为探索LLMs推理能力的极限提供了重要案例研究。 |
| [^2] | [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205) | Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。 |

# 详细

[^1]: 大型语言模型在漏洞检测方面的能力综合研究

    A Comprehensive Study of the Capabilities of Large Language Models for Vulnerability Detection

    [https://arxiv.org/abs/2403.17218](https://arxiv.org/abs/2403.17218)

    本研究调查了十一种领先的大型语言模型在漏洞检测中的能力，并评估了它们的性能，为探索LLMs推理能力的极限提供了重要案例研究。

    

    大型语言模型（LLMs）已经展现出在代码生成和其他软件工程任务方面具有巨大潜力。漏洞检测对于维护软件系统的安全、完整性和可信度至关重要。精确的漏洞检测需要对代码进行推理，这使得它成为探索LLMs推理能力极限的良好案例研究。尽管最近的研究已经利用通用提示技术将LLMs应用于漏洞检测，但它们在这一任务中的完整能力以及在解释确定的漏洞时所犯的错误类型仍不清楚。在本文中，我们调查了十一种领先的在代码生成方面处于最前沿且通常用作编码助手的LLMs，并评估了它们在漏洞检测方面的能力。我们系统地搜索了效果最佳的提示，结合了诸如上下文学习和链式学习等技术。

    arXiv:2403.17218v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated great potential for code generation and other software engineering tasks. Vulnerability detection is of crucial importance to maintaining the security, integrity, and trustworthiness of software systems. Precise vulnerability detection requires reasoning about the code, making it a good case study for exploring the limits of LLMs' reasoning capabilities. Although recent work has applied LLMs to vulnerability detection using generic prompting techniques, their full capabilities for this task and the types of errors they make when explaining identified vulnerabilities remain unclear.   In this paper, we surveyed eleven LLMs that are state-of-the-art in code generation and commonly used as coding assistants, and evaluated their capabilities for vulnerability detection. We systematically searched for the best-performing prompts, incorporating techniques such as in-context learning and chain-of
    
[^2]: Lemur: 使用熵抽样和思维链合并进行日志解析

    Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging

    [https://arxiv.org/abs/2402.18205](https://arxiv.org/abs/2402.18205)

    Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。

    

    大型软件系统产生的日志对监视系统行为至关重要。先进的日志分析有助于检测、报警和诊断系统故障。日志解析是日志分析自动化的关键阶段，它涉及将原始日志消息转换为结构化模板。现有的日志解析器由于依赖于人工制定的规则而无法识别正确的模板。此外，这些方法侧重于统计特征，而忽略了日志消息中的语义信息。为了解决这些挑战，我们提出了一种先进的日志解析框架，采用熵抽样和思维链合并（Lemur）。具体而言，为了摆脱繁琐的手动规则，我们提出了一种受信息熵启发的新型抽样方法，能够有效地对典型日志进行聚类。此外，为了增强日志模板的合并，我们设计了一种思维链方法。

    arXiv:2402.18205v1 Announce Type: cross  Abstract: Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which entails transforming raw log messages into structured templates, constitutes a critical phase in the automation of log analytics. Existing log parsers fail to identify the correct templates due to reliance on human-made rules. Besides, These methods focus on statistical features while ignoring semantic information in log messages. To address these challenges, we introduce a cutting-edge \textbf{L}og parsing framework with \textbf{E}ntropy sampling and Chain-of-Thought \textbf{M}erging (Lemur). Specifically, to discard the tedious manual rules. We propose a novel sampling method inspired by information entropy, which efficiently clusters typical logs. Furthermore, to enhance the merging of log templates, we design a chain-of-thought method f
    

