# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs.](http://arxiv.org/abs/2308.11914) | 通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。 |

# 详细

[^1]: 迈向因果GPT：通过促进LLMs中的因果一致性，基于多智能体的方法实现忠实的知识推理

    Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs. (arXiv:2308.11914v1 [cs.AI])

    [http://arxiv.org/abs/2308.11914](http://arxiv.org/abs/2308.11914)

    通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。

    

    尽管LLMs的发展取得了一些进展，但基于知识的推理仍然是一个长期存在的问题，这是由于知识回忆和推理的脆弱性引起的。现有方法主要通过鼓励LLMs自主计划和解决问题或广泛采样推理链来解决这个问题，但未能解决概念和推理谬误。为了减少推理谬误，我们从多智能体协作中得到启发，提出了一个框架来增加基于知识的推理的忠实度和因果性。具体而言，我们建议使用多个智能体（即推理器和因果评估器）在推理和一致性范式中协作工作，以提高推理的忠实度。推理器专注于提供具有人类因果关系的解决方案，用于解决开放领域的问题。另一方面，因果评估器代理检查解决方案中的答案是否从问题中因果推导出来，反之亦然，并用一个反事实的答案来替代。

    Despite advancements in LLMs, knowledge-based reasoning remains a longstanding issue due to the fragility of knowledge recall and inference. Existing methods primarily encourage LLMs to autonomously plan and solve problems or to extensively sample reasoning chains without addressing the conceptual and inferential fallacies. Attempting to alleviate inferential fallacies and drawing inspiration from multi-agent collaboration, we present a framework to increase faithfulness and causality for knowledge-based reasoning. Specifically, we propose to employ multiple intelligent agents (i.e., reasoner and causal evaluator) to work collaboratively in a reasoning-and-consensus paradigm for elevated reasoning faithfulness. The reasoners focus on providing solutions with human-like causality to solve open-domain problems. On the other hand, the causal evaluator agent scrutinizes if the answer in a solution is causally deducible from the question and vice versa, with a counterfactual answer replacin
    

