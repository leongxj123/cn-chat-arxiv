# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205) | Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。 |

# 详细

[^1]: Lemur: 使用熵抽样和思维链合并进行日志解析

    Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging

    [https://arxiv.org/abs/2402.18205](https://arxiv.org/abs/2402.18205)

    Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。

    

    大型软件系统产生的日志对监视系统行为至关重要。先进的日志分析有助于检测、报警和诊断系统故障。日志解析是日志分析自动化的关键阶段，它涉及将原始日志消息转换为结构化模板。现有的日志解析器由于依赖于人工制定的规则而无法识别正确的模板。此外，这些方法侧重于统计特征，而忽略了日志消息中的语义信息。为了解决这些挑战，我们提出了一种先进的日志解析框架，采用熵抽样和思维链合并（Lemur）。具体而言，为了摆脱繁琐的手动规则，我们提出了一种受信息熵启发的新型抽样方法，能够有效地对典型日志进行聚类。此外，为了增强日志模板的合并，我们设计了一种思维链方法。

    arXiv:2402.18205v1 Announce Type: cross  Abstract: Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which entails transforming raw log messages into structured templates, constitutes a critical phase in the automation of log analytics. Existing log parsers fail to identify the correct templates due to reliance on human-made rules. Besides, These methods focus on statistical features while ignoring semantic information in log messages. To address these challenges, we introduce a cutting-edge \textbf{L}og parsing framework with \textbf{E}ntropy sampling and Chain-of-Thought \textbf{M}erging (Lemur). Specifically, to discard the tedious manual rules. We propose a novel sampling method inspired by information entropy, which efficiently clusters typical logs. Furthermore, to enhance the merging of log templates, we design a chain-of-thought method f
    

