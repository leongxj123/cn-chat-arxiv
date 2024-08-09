# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Distributional Uncertainty of the SHAP score in Explainable Machine Learning.](http://arxiv.org/abs/2401.12731) | 本研究提出了一个原则性框架，用于处理在未知实体群体分布下的SHAP评分问题。通过考虑一个不确定性区域，我们可以确定所有特征的SHAP评分的紧束范围。 |
| [^2] | [A Language-Agent Approach to Formal Theorem-Proving.](http://arxiv.org/abs/2310.04353) | COPRA是一种面向形式定理证明的语言代理方法，利用大型语言模型进行上下文学习，通过选择策略和检索定义和引理进行证明，在MiniF2F基准和Coq任务上表现出优异的性能。 |
| [^3] | [DR-HAI: Argumentation-based Dialectical Reconciliation in Human-AI Interactions.](http://arxiv.org/abs/2306.14694) | DR-HAI是一个新颖的基于论证的框架，旨在通过互动调和解决人工智能与人类之间的知识差异，为促进有效的人工智能与人类交互提供了一个有希望的方向。 |

# 详细

[^1]: SHAP评分在可解释机器学习中的分布不确定性

    The Distributional Uncertainty of the SHAP score in Explainable Machine Learning. (arXiv:2401.12731v1 [cs.AI])

    [http://arxiv.org/abs/2401.12731](http://arxiv.org/abs/2401.12731)

    本研究提出了一个原则性框架，用于处理在未知实体群体分布下的SHAP评分问题。通过考虑一个不确定性区域，我们可以确定所有特征的SHAP评分的紧束范围。

    

    归属分数反映了输入实体中的特征值对机器学习模型输出的重要性。其中最受欢迎的评分之一是SHAP评分，它是合作博弈理论中Shapley值的具体实例。该评分的定义依赖于实体群体的概率分布。由于通常不知道精确的分布，因此需要主观地进行分配或从数据中进行估计，这可能会导致误导性的特征评分。在本文中，我们提出了一个基于不知道实体群体分布的SHAP评分推理的原则性框架。在我们的框架中，我们考虑一个包含潜在分布的不确定性区域，而特征的SHAP评分成为在该区域上定义的一个函数。我们研究了找到该函数的最大值和最小值的基本问题，这使我们能够确定所有特征的SHAP评分的紧束范围。

    Attribution scores reflect how important the feature values in an input entity are for the output of a machine learning model. One of the most popular attribution scores is the SHAP score, which is an instantiation of the general Shapley value used in coalition game theory. The definition of this score relies on a probability distribution on the entity population. Since the exact distribution is generally unknown, it needs to be assigned subjectively or be estimated from data, which may lead to misleading feature scores. In this paper, we propose a principled framework for reasoning on SHAP scores under unknown entity population distributions. In our framework, we consider an uncertainty region that contains the potential distributions, and the SHAP score of a feature becomes a function defined over this region. We study the basic problems of finding maxima and minima of this function, which allows us to determine tight ranges for the SHAP scores of all features. In particular, we pinp
    
[^2]: 一种面向形式定理证明的语言代理方法

    A Language-Agent Approach to Formal Theorem-Proving. (arXiv:2310.04353v1 [cs.LG])

    [http://arxiv.org/abs/2310.04353](http://arxiv.org/abs/2310.04353)

    COPRA是一种面向形式定理证明的语言代理方法，利用大型语言模型进行上下文学习，通过选择策略和检索定义和引理进行证明，在MiniF2F基准和Coq任务上表现出优异的性能。

    

    语言代理是利用大型语言模型（LLM）进行上下文学习来与外部环境进行交互的方法，最近被认为是一种有前景的控制任务方法。

    Language agents, which use a large language model (LLM) capable of in-context learning to interact with an external environment, have recently emerged as a promising approach to control tasks. We present the first language-agent approach to formal theorem-proving. Our method, COPRA, uses a high-capacity, black-box LLM (GPT-4) as part of a policy for a stateful backtracking search. During the search, the policy can select proof tactics and retrieve lemmas and definitions from an external database. Each selected tactic is executed in the underlying proof framework, and the execution feedback is used to build the prompt for the next policy invocation. The search also tracks selected information from its history and uses it to reduce hallucinations and unnecessary LLM queries.  We evaluate COPRA on the miniF2F benchmark for Lean and a set of Coq tasks from the Compcert project. On these benchmarks, COPRA is significantly better than one-shot invocations of GPT-4, as well as state-of-the-ar
    
[^3]: DR-HAI: 人工智能与人类交互中基于论证的辩证调和

    DR-HAI: Argumentation-based Dialectical Reconciliation in Human-AI Interactions. (arXiv:2306.14694v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2306.14694](http://arxiv.org/abs/2306.14694)

    DR-HAI是一个新颖的基于论证的框架，旨在通过互动调和解决人工智能与人类之间的知识差异，为促进有效的人工智能与人类交互提供了一个有希望的方向。

    

    我们提出了DR-HAI，这是一个新颖的基于论证的框架，旨在扩展人类感知规划中常用的模型调和方法，以增强人工智能与人类的交互。通过采用基于论证的对话范式，DR-HAI能够进行互动调和，解决解释者和被解释者之间的知识差异。我们对DR-HAI的操作语义进行了形式化描述，提供了理论保证，并对其效果进行了经验评估。我们的研究结果表明，DR-HAI为促进有效的人工智能与人类交互提供了一个具有潜力的方向。

    We present DR-HAI -- a novel argumentation-based framework designed to extend model reconciliation approaches, commonly used in human-aware planning, for enhanced human-AI interaction. By adopting an argumentation-based dialogue paradigm, DR-HAI enables interactive reconciliation to address knowledge discrepancies between an explainer and an explainee. We formally describe the operational semantics of DR-HAI, provide theoretical guarantees, and empirically evaluate its efficacy. Our findings suggest that DR-HAI offers a promising direction for fostering effective human-AI interactions.
    

