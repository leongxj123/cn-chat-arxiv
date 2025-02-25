# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tur[k]ingBench: A Challenge Benchmark for Web Agents](https://arxiv.org/abs/2403.11905) | Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。 |
| [^2] | [KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents](https://arxiv.org/abs/2403.03101) | KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。 |

# 详细

[^1]: Tur[k]ingBench：用于网络代理的挑战基准测试

    Tur[k]ingBench: A Challenge Benchmark for Web Agents

    [https://arxiv.org/abs/2403.11905](https://arxiv.org/abs/2403.11905)

    Tur[k]ingBench是一个挑战性的网络代理基准测试，用于评估最先进的多模态模型在处理包含文本指示和多模态上下文的复杂任务时的泛化能力。

    

    最近的聊天机器人展示了在原始文本形式下理解和交流的令人印象深刻的能力。然而，世界上不仅仅是原始文本。例如，人们在网页上花费大量时间，在这些网页上，文本与其他形式交织在一起，并以各种复杂互动的形式完成任务。最先进的多模型是否能够推广到这种复杂的领域呢？为了回答这个问题，我们介绍了TurkingBench，一个由包含多模态背景的文本说明制定的任务基准。与现有的使用人工合成的网页的工作不同，这里我们使用最初设计用于各种注释目的的自然HTML页面。每个任务的HTML说明也被实例化为各种值（从众包任务获得）以形成任务的新实例。这个基准包含32.2K个实例。

    arXiv:2403.11905v1 Announce Type: new  Abstract: Recent chatbots have demonstrated impressive ability to understand and communicate in raw-text form. However, there is more to the world than raw text. For example, humans spend long hours of their time on web pages, where text is intertwined with other modalities and tasks are accomplished in the form of various complex interactions. Can state-of-the-art multi-modal models generalize to such complex domains?   To address this question, we introduce TurkingBench, a benchmark of tasks formulated as web pages containing textual instructions with multi-modal context. Unlike existing work which employs artificially synthesized web pages, here we use natural HTML pages that were originally designed for crowdsourcing workers for various annotation purposes. The HTML instructions of each task are also instantiated with various values (obtained from the crowdsourcing tasks) to form new instances of the task. This benchmark contains 32.2K instanc
    
[^2]: KnowAgent: 知识增强规划用于基于LLM的Agent

    KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents

    [https://arxiv.org/abs/2403.03101](https://arxiv.org/abs/2403.03101)

    KnowAgent引入了显式动作知识，通过动作知识库和知识型自学习策略来增强LLM的规划能力，从而改善语言Agent的规划表现。

    

    大型语言模型(LLMs)在复杂推理任务中表现出巨大潜力，但在处理更复杂的挑战时仍有所不足，特别是与环境互动通过生成可执行动作时。这种不足主要来自于语言Agent中缺乏内置动作知识，导致在任务求解过程中无法有效引导规划轨迹，从而导致规划幻觉。为了解决这个问题，我们引入了KnowAgent，一种旨在通过整合显式动作知识来增强LLM规划能力的新方法。具体而言，KnowAgent采用了一个动作知识库和一个知识型自学习策略来限制规划过程中的行动路径，实现更合理的轨迹合成，进而提高语言Agent的计划性能。基于HotpotQA和ALFWorld的实验结果基于不同的主干模型。

    arXiv:2403.03101v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone m
    

