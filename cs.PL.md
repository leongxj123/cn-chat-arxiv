# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Language-Agent Approach to Formal Theorem-Proving.](http://arxiv.org/abs/2310.04353) | COPRA是一种面向形式定理证明的语言代理方法，利用大型语言模型进行上下文学习，通过选择策略和检索定义和引理进行证明，在MiniF2F基准和Coq任务上表现出优异的性能。 |

# 详细

[^1]: 一种面向形式定理证明的语言代理方法

    A Language-Agent Approach to Formal Theorem-Proving. (arXiv:2310.04353v1 [cs.LG])

    [http://arxiv.org/abs/2310.04353](http://arxiv.org/abs/2310.04353)

    COPRA是一种面向形式定理证明的语言代理方法，利用大型语言模型进行上下文学习，通过选择策略和检索定义和引理进行证明，在MiniF2F基准和Coq任务上表现出优异的性能。

    

    语言代理是利用大型语言模型（LLM）进行上下文学习来与外部环境进行交互的方法，最近被认为是一种有前景的控制任务方法。

    Language agents, which use a large language model (LLM) capable of in-context learning to interact with an external environment, have recently emerged as a promising approach to control tasks. We present the first language-agent approach to formal theorem-proving. Our method, COPRA, uses a high-capacity, black-box LLM (GPT-4) as part of a policy for a stateful backtracking search. During the search, the policy can select proof tactics and retrieve lemmas and definitions from an external database. Each selected tactic is executed in the underlying proof framework, and the execution feedback is used to build the prompt for the next policy invocation. The search also tracks selected information from its history and uses it to reduce hallucinations and unnecessary LLM queries.  We evaluate COPRA on the miniF2F benchmark for Lean and a set of Coq tasks from the Compcert project. On these benchmarks, COPRA is significantly better than one-shot invocations of GPT-4, as well as state-of-the-ar
    

