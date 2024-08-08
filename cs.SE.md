# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences](https://arxiv.org/abs/2403.09032) | 介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。 |

# 详细

[^1]: CodeUltraFeedback：一种用于将大型语言模型与编程偏好对齐的LLM作为法官数据集

    CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences

    [https://arxiv.org/abs/2403.09032](https://arxiv.org/abs/2403.09032)

    介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。

    

    评估大型语言模型（LLMs）与用户定义的编程偏好的对齐性是一项具有挑战性的工作，需要评估复杂文本LLMs的输出。现有基准仰赖自动化指标和静态分析工具，未能评估用户指令和LLM输出中的微妙之处，突显了对LLM偏好对齐的大规模数据集和基准的需求。在本文中，我们介绍了CodeUltraFeedback，一个包含10,000个复杂指令的偏好数据集，通过AI反馈来调整和对齐LLMs与编程偏好。我们使用14种不同的LLMs对这些指令生成响应，然后根据它们与五种编程偏好的对齐情况进行注释，使用GPT-3.5的LLM作为法官方法产生数字和文本反馈。我们还提出了CODAL-Bench，一个用于评估LLM与这些编程偏好对齐的基准。我们的结果显示C

    arXiv:2403.09032v1 Announce Type: cross  Abstract: Evaluating the alignment of large language models (LLMs) with user-defined coding preferences is a challenging endeavour that requires assessing intricate textual LLMs' outputs. By relying on automated metrics and static analysis tools, existing benchmarks fail to assess nuances in user instructions and LLM outputs, highlighting the need for large-scale datasets and benchmarks for LLM preference alignment. In this paper, we introduce CodeUltraFeedback, a preference dataset of 10,000 complex instructions to tune and align LLMs to coding preferences through AI feedback. We generate responses to the instructions using a pool of 14 diverse LLMs, which we then annotate according to their alignment with five coding preferences using the LLM-as-a-Judge approach with GPT-3.5, producing both numerical and textual feedback. We also present CODAL-Bench, a benchmark for assessing LLM alignment with these coding preferences. Our results show that C
    

