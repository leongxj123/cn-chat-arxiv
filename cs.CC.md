# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) | 思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。 |

# 详细

[^1]: 思维链激发变压器解决固有串行问题的能力

    Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

    [https://arxiv.org/abs/2402.12875](https://arxiv.org/abs/2402.12875)

    思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。

    

    指导模型生成一系列中间步骤，即思维链（CoT），是提高大型语言模型（LLMs）在算术和符号推理任务上准确性的高效方法。然而，CoT背后的机制仍不清楚。这项工作通过表达性的视角提供了对解码器专用变压器的CoT能力的理论理解。在概念上，CoT赋予模型执行固有串行计算的能力，而这种能力在变压器中缺乏，特别是当深度较低时。先前的作品已经表明，在没有CoT的情况下，具有有限精度$\mathsf{poly}(n)$嵌入尺寸的恒定深度变压器只能在$\mathsf{TC}^0$中解决问题。我们首先展示了具有常数位精度的恒定深度变压器的更紧密的表达性上界，它只能解决$\mathsf{AC}^0$中的问题。

    arXiv:2402.12875v1 Announce Type: new  Abstract: Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks. However, the mechanism behind CoT remains unclear. This work provides a theoretical understanding of the power of CoT for decoder-only transformers through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length $n$, previous works have shown that constant-depth transformers with finite precision $\mathsf{poly}(n)$ embedding size can only solve problems in $\mathsf{TC}^0$ without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in $\mathsf{AC}^0$, a 
    

