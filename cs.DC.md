# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789) | FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用 |

# 详细

[^1]: FlexLLM：一种用于共同提供大型语言模型推理和参数高效微调的系统

    FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2402.18789](https://arxiv.org/abs/2402.18789)

    FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用

    

    Parameter-efficient finetuning（PEFT）是一种广泛使用的技术，用于为不同任务调整大型语言模型。通常，服务提供商会为用户创建单独的系统，以执行PEFT模型微调和推理任务。这是因为现有系统无法处理包含推理和PEFT微调请求混合的工作负载。因此，共享的GPU资源利用不足，导致效率低下。为解决这一问题，我们提出了FlexLLM，这是第一个可以在同一迭代中为推理和参数高效微调请求提供服务的系统。我们的系统利用这两个任务的互补性质，并利用共享的GPU资源来共同运行它们，使用一种称为共同提供的方法。为实现这一目标，FlexLLM引入了一种新颖的标记级微调机制，将序列的微调计算分解为更小的标记级计算，并使用依赖并行化。

    arXiv:2402.18789v1 Announce Type: cross  Abstract: Parameter-efficient finetuning (PEFT) is a widely used technique to adapt large language models for different tasks. Service providers typically create separate systems for users to perform PEFT model finetuning and inference tasks. This is because existing systems cannot handle workloads that include a mix of inference and PEFT finetuning requests. As a result, shared GPU resources are underutilized, leading to inefficiencies. To address this problem, we present FlexLLM, the first system that can serve inference and parameter-efficient finetuning requests in the same iteration. Our system leverages the complementary nature of these two tasks and utilizes shared GPU resources to run them jointly, using a method called co-serving. To achieve this, FlexLLM introduces a novel token-level finetuning mechanism, which breaks down the finetuning computation of a sequence into smaller token-level computations and uses dependent parallelization
    

