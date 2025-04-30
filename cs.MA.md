# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Era of Semantic Decoding](https://arxiv.org/abs/2403.14562) | 提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。 |
| [^2] | [Evaluating Multi-Agent Coordination Abilities in Large Language Models.](http://arxiv.org/abs/2310.03903) | 本研究构建了使用大型语言模型（LLMs）的智能体，并评估其在多智能体协调中的有效性。我们引入了LLM-Co框架，用于在三个游戏环境中评估LLMs的协调能力。评估结果显示LLMs具有推断伙伴意图和理解其行动的能力。 |

# 详细

[^1]: 语义解码时代

    The Era of Semantic Decoding

    [https://arxiv.org/abs/2403.14562](https://arxiv.org/abs/2403.14562)

    提出了一种名为语义解码的新观点，将LLM、人类输入和各种工具之间的协作过程构建为语义空间中的优化过程，促进了高效输出的构建。

    

    最近的研究展现了在LLM（大型语言模型）、人类输入和各种工具之间编排协作以解决LLM固有局限性的想法具有巨大潜力。我们提出了一个名为语义解码的新观点，将这些协作过程构建为语义空间中的优化过程。具体来说，我们将LLM概念化为操纵我们称之为语义标记（已知思想）的有意义信息片段的语义处理器。LLM是众多其他语义处理器之一，包括人类和工具，比如搜索引擎或代码执行器。语义处理器集体参与语义标记的动态交流，逐步构建高效输出。我们称这些在语义空间中进行优化和搜索的协同作用，为语义解码算法。这个概念与已广为研究的语义解码问题直接平行。

    arXiv:2403.14562v1 Announce Type: cross  Abstract: Recent work demonstrated great promise in the idea of orchestrating collaborations between LLMs, human input, and various tools to address the inherent limitations of LLMs. We propose a novel perspective called semantic decoding, which frames these collaborative processes as optimization procedures in semantic space. Specifically, we conceptualize LLMs as semantic processors that manipulate meaningful pieces of information that we call semantic tokens (known thoughts). LLMs are among a large pool of other semantic processors, including humans and tools, such as search engines or code executors. Collectively, semantic processors engage in dynamic exchanges of semantic tokens to progressively construct high-utility outputs. We refer to these orchestrated interactions among semantic processors, optimizing and searching in semantic space, as semantic decoding algorithms. This concept draws a direct parallel to the well-studied problem of s
    
[^2]: 在大型语言模型中评估多智能体协调能力

    Evaluating Multi-Agent Coordination Abilities in Large Language Models. (arXiv:2310.03903v1 [cs.CL])

    [http://arxiv.org/abs/2310.03903](http://arxiv.org/abs/2310.03903)

    本研究构建了使用大型语言模型（LLMs）的智能体，并评估其在多智能体协调中的有效性。我们引入了LLM-Co框架，用于在三个游戏环境中评估LLMs的协调能力。评估结果显示LLMs具有推断伙伴意图和理解其行动的能力。

    

    当代人工智能研究的一个重要目标是开发能够熟练进行多智能体协调、有效与人类和其他系统合作的智能体。大型语言模型（LLM）以其显著的理解、生成和解释语言的能力成为开发这种智能体的有希望的候选模型。本研究中，我们构建了使用LLM构建的智能体，并评估其在各种协调场景中的有效性。我们引入了特别设计的LLM-Co框架，使LLM能够参与协调游戏。通过LLM-Co框架，我们在三个游戏环境中进行评估，并将评估分为五个方面：心智理论、情境推理、持续协调、对合作伙伴的稳健性和明确辅助。首先，心智理论和情境推理的评估揭示了LLM推断伙伴意图和理解其行动的能力。

    A pivotal aim in contemporary AI research is to develop agents proficient in multi-agent coordination, enabling effective collaboration with both humans and other systems. Large Language Models (LLMs), with their notable ability to understand, generate, and interpret language in a human-like manner, stand out as promising candidates for the development of such agents. In this study, we build and assess the effectiveness of agents crafted using LLMs in various coordination scenarios. We introduce the LLM-Coordination (LLM-Co) Framework, specifically designed to enable LLMs to play coordination games. With the LLM-Co framework, we conduct our evaluation with three game environments and organize the evaluation into five aspects: Theory of Mind, Situated Reasoning, Sustained Coordination, Robustness to Partners, and Explicit Assistance. First, the evaluation of the Theory of Mind and Situated Reasoning reveals the capabilities of LLM to infer the partner's intention and reason actions acco
    

