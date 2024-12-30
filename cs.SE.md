# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences](https://arxiv.org/abs/2403.09032) | 介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。 |
| [^2] | [Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models.](http://arxiv.org/abs/2308.10462) | 本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。 |

# 详细

[^1]: CodeUltraFeedback：一种用于将大型语言模型与编程偏好对齐的LLM作为法官数据集

    CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences

    [https://arxiv.org/abs/2403.09032](https://arxiv.org/abs/2403.09032)

    介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。

    

    评估大型语言模型（LLMs）与用户定义的编程偏好的对齐性是一项具有挑战性的工作，需要评估复杂文本LLMs的输出。现有基准仰赖自动化指标和静态分析工具，未能评估用户指令和LLM输出中的微妙之处，突显了对LLM偏好对齐的大规模数据集和基准的需求。在本文中，我们介绍了CodeUltraFeedback，一个包含10,000个复杂指令的偏好数据集，通过AI反馈来调整和对齐LLMs与编程偏好。我们使用14种不同的LLMs对这些指令生成响应，然后根据它们与五种编程偏好的对齐情况进行注释，使用GPT-3.5的LLM作为法官方法产生数字和文本反馈。我们还提出了CODAL-Bench，一个用于评估LLM与这些编程偏好对齐的基准。我们的结果显示C

    arXiv:2403.09032v1 Announce Type: cross  Abstract: Evaluating the alignment of large language models (LLMs) with user-defined coding preferences is a challenging endeavour that requires assessing intricate textual LLMs' outputs. By relying on automated metrics and static analysis tools, existing benchmarks fail to assess nuances in user instructions and LLM outputs, highlighting the need for large-scale datasets and benchmarks for LLM preference alignment. In this paper, we introduce CodeUltraFeedback, a preference dataset of 10,000 complex instructions to tune and align LLMs to coding preferences through AI feedback. We generate responses to the instructions using a pool of 14 diverse LLMs, which we then annotate according to their alignment with five coding preferences using the LLM-as-a-Judge approach with GPT-3.5, producing both numerical and textual feedback. We also present CODAL-Bench, a benchmark for assessing LLM alignment with these coding preferences. Our results show that C
    
[^2]: 探索大语言模型用于代码生成的参数高效微调技术

    Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models. (arXiv:2308.10462v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2308.10462](http://arxiv.org/abs/2308.10462)

    本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。

    

    大型语言模型（LLM）展示了在没有特定微调的情况下，即可根据自然语言意图生成准确的代码片段的印象能力。尽管先前的研究已经突出了微调LLMs的优势，但这个过程代价高，对于拥有数十亿个参数的模型来说，在资源稀缺的环境下是不切实际的。为了解决这些挑战，以前的研究探索了在上下文学习（ICL）作为一种策略，用任务特定的提示示例指导LLM生成过程。然而，ICL引入了一些不便之处，比如需要设计上下文相关的提示和没有学习任务特定的参数，从而限制了下游任务的性能。在这种情况下，我们预见参数高效微调（PEFT）技术作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将LLM专门用于任务特定的数据。

    Large Language Models (LLMs) demonstrate impressive capabilities to generate accurate code snippets given natural language intents in zero-shot, i.e., without the need for specific fine-tuning. While prior studies have highlighted the advantages of fine-tuning LLMs, this process incurs high computational costs, making it impractical in resource-scarce environments, particularly for models with billions of parameters. To address these challenges, previous research explored In-Context Learning (ICL) as a strategy to guide the LLM generative process with task-specific prompt examples. However, ICL introduces inconveniences, such as the need for designing contextually relevant prompts and the absence of learning task-specific parameters, thereby limiting downstream task performance. In this context, we foresee Parameter-Efficient Fine-Tuning (PEFT) techniques as a promising approach to efficiently specialize LLMs to task-specific data while maintaining reasonable resource consumption. In t
    

