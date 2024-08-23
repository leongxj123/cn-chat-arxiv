# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) | 将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。 |

# 详细

[^1]: 作为可优化图的语言代理

    Language Agents as Optimizable Graphs

    [https://arxiv.org/abs/2402.16823](https://arxiv.org/abs/2402.16823)

    将基于LLM的代理统一描述为计算图，提出新颖的自动图优化器来改进节点和边，实现了代理之间的自动协作和改进。

    

    多种人类设计的提升技术被提出，用于改进基于大型语言模型（LLMs）的问题求解器，产生了许多不同的代码库。我们通过将LLM代理描述为计算图来统一这些方法。节点实现处理多模态数据或查询LLMs的功能，并且边描述操作之间的信息流动。图形可以递归地组合成代表不同代理之间协作层次的更大组合图（其中边连接不同代理的操作）。我们的新颖自动图优化器（1）优化节点级LLM提示（节点优化）并（2）通过改变图连接性来改善代理协调（边缘优化）。实验证明我们的框架可用于高效开发、集成和自动改进各种LLM代理。代码可在https://github.com/metauto-ai/gptswarm找到。

    arXiv:2402.16823v1 Announce Type: cross  Abstract: Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches by describing LLM-based agents as computational graphs. The nodes implement functions to process multimodal data or query LLMs, and the edges describe the information flow between operations. Graphs can be recursively combined into larger composite graphs representing hierarchies of inter-agent collaboration (where edges connect operations of different agents). Our novel automatic graph optimizers (1) refine node-level LLM prompts (node optimization) and (2) improve agent orchestration by changing graph connectivity (edge optimization). Experiments demonstrate that our framework can be used to efficiently develop, integrate, and automatically improve various LLM agents. The code can be found at https://github.com/metauto-ai/gptswarm.
    

