# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SymbolicAI: A framework for logic-based approaches combining generative models and solvers](https://arxiv.org/abs/2402.00854) | SymbolicAI是一个基于逻辑的框架，将生成模型与多种求解器无缝集成，通过将大型语言模型作为语义解析器，实现了符号推理与生成式人工智能的融合。 |
| [^2] | [MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks.](http://arxiv.org/abs/2312.15960) | MoTCoder是一个使用思维模块提升大型语言模型在挑战性编程任务中能力的框架，通过创新的指令调整促进任务的分解和模块化，显著提高生成解决方案的准确性和模块化程度。 |

# 详细

[^1]: SymbolicAI: 一个结合生成模型和求解器的基于逻辑的方法的框架

    SymbolicAI: A framework for logic-based approaches combining generative models and solvers

    [https://arxiv.org/abs/2402.00854](https://arxiv.org/abs/2402.00854)

    SymbolicAI是一个基于逻辑的框架，将生成模型与多种求解器无缝集成，通过将大型语言模型作为语义解析器，实现了符号推理与生成式人工智能的融合。

    

    我们介绍了SymbolicAI，这是一个多功能且模块化的框架，采用基于逻辑的方法来处理生成过程中的概念学习和流程管理。SymbolicAI通过将大型语言模型（LLM）作为语义解析器来执行基于自然语言和形式语言指令的任务，从而弥合了符号推理和生成式人工智能之间的差距，使生成模型与各种求解器无缝集成。我们利用概率编程原理来处理复杂任务，并利用可微分和经典编程范 paradigms 的各自优势。该框架引入了一系列多态的、组合的和自指的数据流操作，将LLM的输出与用户的目标对齐。因此，我们可以在具有零次和少次学习能力的各种基础模型之间进行过渡，并与擅长解决特定问题的专业化调优模型或求解器配合使用。

    We introduce SymbolicAI, a versatile and modular framework employing a logic-based approach to concept learning and flow management in generative processes. SymbolicAI enables the seamless integration of generative models with a diverse range of solvers by treating large language models (LLMs) as semantic parsers that execute tasks based on both natural and formal language instructions, thus bridging the gap between symbolic reasoning and generative AI. We leverage probabilistic programming principles to tackle complex tasks, and utilize differentiable and classical programming paradigms with their respective strengths. The framework introduces a set of polymorphic, compositional, and self-referential operations for data stream manipulation, aligning LLM outputs with user objectives. As a result, we can transition between the capabilities of various foundation models endowed with zero- and few-shot learning capabilities and specialized, fine-tuned models or solvers proficient in addres
    
[^2]: MoTCoder: 使用思维模块提升大型语言模型在具有挑战性的编程任务中的能力。

    MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks. (arXiv:2312.15960v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15960](http://arxiv.org/abs/2312.15960)

    MoTCoder是一个使用思维模块提升大型语言模型在挑战性编程任务中能力的框架，通过创新的指令调整促进任务的分解和模块化，显著提高生成解决方案的准确性和模块化程度。

    

    大型语言模型(LLMs)在处理简单的编程任务方面展示出了令人印象深刻的能力。然而，当面对更具挑战性的编程问题时，它们的性能往往表现不佳。我们观察到传统模型往往生成作为单一代码块的解决方案，限制了它们在解决复杂问题上的有效性。为了克服这个限制，我们提出了Modular-of-Thought Coder (MoTCoder)。我们引入了一种创新的MoT指令调整框架，旨在促进将任务分解为逻辑子任务和子模块。我们的研究发现，通过培养和利用子模块，MoTCoder显著提高了生成解决方案的模块化和正确性，导致在APPS上相对pass@1改进了12.9%，在CodeContests上相对pass@1改进了9.43%。我们的代码可在https://github.com/dvlab-research/MoTCoder获得。

    Large Language Models (LLMs) have showcased impressive capabilities in handling straightforward programming tasks. However, their performance tends to falter when confronted with more challenging programming problems. We observe that conventional models often generate solutions as monolithic code blocks, restricting their effectiveness in tackling intricate questions. To overcome this limitation, we present Modular-of-Thought Coder (MoTCoder). We introduce a pioneering framework for MoT instruction tuning, designed to promote the decomposition of tasks into logical sub-tasks and sub-modules. Our investigations reveal that, through the cultivation and utilization of sub-modules, MoTCoder significantly improves both the modularity and correctness of the generated solutions, leading to substantial relative pass@1 improvements of 12.9% on APPS and 9.43% on CodeContests. Our codes are available at https://github.com/dvlab-research/MoTCoder.
    

