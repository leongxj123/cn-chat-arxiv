# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Closure Discovery for Coarse-Grained Partial Differential Equations using Multi-Agent Reinforcement Learning](https://rss.arxiv.org/abs/2402.00972) | 使用多智能体强化学习(MARL)识别未精细解析的PDEs中的闭合项，通过部署中央策略和卷积神经网络(CNN)，能够准确预测和加速模拟。 |
| [^2] | [Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs.](http://arxiv.org/abs/2308.11914) | 通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。 |

# 详细

[^1]: 通过多智能体强化学习识别粗粒度偏微分方程的闭合项

    Closure Discovery for Coarse-Grained Partial Differential Equations using Multi-Agent Reinforcement Learning

    [https://rss.arxiv.org/abs/2402.00972](https://rss.arxiv.org/abs/2402.00972)

    使用多智能体强化学习(MARL)识别未精细解析的PDEs中的闭合项，通过部署中央策略和卷积神经网络(CNN)，能够准确预测和加速模拟。

    

    可靠地预测天气、野火和流行病等关键现象通常基于由偏微分方程(PDEs)描述的模型。然而，捕捉这种PDEs中全面的时空尺度范围的模拟通常是代价高昂的。因此，通常会使用利用启发式方法和经验闭合项的粗粒度模拟作为替代方法。我们提出了一种通过多智能体强化学习(MARL)识别未精细解析的PDEs中闭合项的新颖和系统的方法。MARL的形式化结合了归纳偏差，并利用部署了由卷积神经网络(CNN)高效表示的中央策略来利用局部性。通过对对流方程和Burgers方程的数值解进行演示，我们展示了MARL的能力和限制。我们的结果显示，MARL对于内外分布的测试案例可以准确预测，并且与精细解析相比有显著的加速效果。

    Reliable predictions of critical phenomena, such as weather, wildfires and epidemics are often founded on models described by Partial Differential Equations (PDEs). However, simulations that capture the full range of spatio-temporal scales in such PDEs are often prohibitively expensive. Consequently, coarse-grained simulations that employ heuristics and empirical closure terms are frequently utilized as an alternative. We propose a novel and systematic approach for identifying closures in under-resolved PDEs using Multi-Agent Reinforcement Learning (MARL). The MARL formulation incorporates inductive bias and exploits locality by deploying a central policy represented efficiently by Convolutional Neural Networks (CNN). We demonstrate the capabilities and limitations of MARL through numerical solutions of the advection equation and the Burgers' equation. Our results show accurate predictions for in- and out-of-distribution test cases as well as a significant speedup compared to resolving
    
[^2]: 迈向因果GPT：通过促进LLMs中的因果一致性，基于多智能体的方法实现忠实的知识推理

    Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs. (arXiv:2308.11914v1 [cs.AI])

    [http://arxiv.org/abs/2308.11914](http://arxiv.org/abs/2308.11914)

    通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。

    

    尽管LLMs的发展取得了一些进展，但基于知识的推理仍然是一个长期存在的问题，这是由于知识回忆和推理的脆弱性引起的。现有方法主要通过鼓励LLMs自主计划和解决问题或广泛采样推理链来解决这个问题，但未能解决概念和推理谬误。为了减少推理谬误，我们从多智能体协作中得到启发，提出了一个框架来增加基于知识的推理的忠实度和因果性。具体而言，我们建议使用多个智能体（即推理器和因果评估器）在推理和一致性范式中协作工作，以提高推理的忠实度。推理器专注于提供具有人类因果关系的解决方案，用于解决开放领域的问题。另一方面，因果评估器代理检查解决方案中的答案是否从问题中因果推导出来，反之亦然，并用一个反事实的答案来替代。

    Despite advancements in LLMs, knowledge-based reasoning remains a longstanding issue due to the fragility of knowledge recall and inference. Existing methods primarily encourage LLMs to autonomously plan and solve problems or to extensively sample reasoning chains without addressing the conceptual and inferential fallacies. Attempting to alleviate inferential fallacies and drawing inspiration from multi-agent collaboration, we present a framework to increase faithfulness and causality for knowledge-based reasoning. Specifically, we propose to employ multiple intelligent agents (i.e., reasoner and causal evaluator) to work collaboratively in a reasoning-and-consensus paradigm for elevated reasoning faithfulness. The reasoners focus on providing solutions with human-like causality to solve open-domain problems. On the other hand, the causal evaluator agent scrutinizes if the answer in a solution is causally deducible from the question and vice versa, with a counterfactual answer replacin
    

