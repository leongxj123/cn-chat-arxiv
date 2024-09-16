# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Activation Steering for Robust Type Prediction in CodeLLMs](https://arxiv.org/abs/2404.01903) | 我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。 |
| [^2] | [CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218) | CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。 |

# 详细

[^1]: 在CodeLLMs中实现类型预测的鲁棒激活导向技术

    Activation Steering for Robust Type Prediction in CodeLLMs

    [https://arxiv.org/abs/2404.01903](https://arxiv.org/abs/2404.01903)

    我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。

    

    预训练在代码上的现代LLMs能够成功地完成各种编程任务。然而，它们的性能对语法特征非常敏感，例如变量和类型的名称、代码结构以及类型提示的存在。我们提出了一种推理时技术，使CodeLLMs更能抵御语法干扰因素，这些因素与语义无关。我们的方法依赖于激活导向，涉及编辑内部模型激活以将模型引导到正确的预测。我们通过从突变测试中汲取灵感构建激活向量的新方法，该方法构建最小的破坏语义的代码编辑。相比之下，我们从保留语义的代码编辑中构建激活向量。我们将我们的方法应用于逐渐类型化语言Python和TypeScript的类型预测任务。这种方法可以纠正高达90%的类型错误预测。

    arXiv:2404.01903v1 Announce Type: new  Abstract: Contemporary LLMs pretrained on code are capable of succeeding at a wide variety of programming tasks. However, their performance is very sensitive to syntactic features, such as the names of variables and types, the structure of code, and presence of type hints. We contribute an inference-time technique to make CodeLLMs more robust to syntactic distractors that are semantically irrelevant. Our methodology relies on activation steering, which involves editing internal model activations to steer the model towards the correct prediction. We contribute a novel way to construct steering vectors by taking inspiration from mutation testing, which constructs minimal semantics-breaking code edits. In contrast, we construct steering vectors from semantics-preserving code edits. We apply our approach to the task of type prediction for the gradually typed languages Python and TypeScript. This approach corrects up to 90% of type mispredictions. Fina
    
[^2]: CoverUp：基于覆盖率引导的LLM测试生成系统

    CoverUp: Coverage-Guided LLM-Based Test Generation

    [https://arxiv.org/abs/2403.16218](https://arxiv.org/abs/2403.16218)

    CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。

    

    本文介绍了CoverUp，这是一个新型系统，通过覆盖率分析和大型语言模型（LLM）的结合驱动生成高覆盖率的Python回归测试。CoverUp通过迭代改善覆盖率，将覆盖率分析与LLM对话交替进行，以便将注意力集中在尚未涵盖的代码行和分支上。最终的测试套件相比当前技术水平显著提高了覆盖率：与CodaMosa相比，一种混合LLM / 基于搜索的软件测试系统，CoverUp在各方面都大幅提高了覆盖率。以模块为基础，CoverUp实现了81%的中位线覆盖率（对比62%）、53%的分支覆盖率（对比35%）和78%的线+分支覆盖率（对比55%）。我们展示了CoverUp的迭代、覆盖率引导方法对其有效性至关重要，为其成功的近一半作出了贡献。

    arXiv:2403.16218v1 Announce Type: cross  Abstract: This paper presents CoverUp, a novel system that drives the generation of high-coverage Python regression tests via a combination of coverage analysis and large-language models (LLMs). CoverUp iteratively improves coverage, interleaving coverage analysis with dialogs with the LLM to focus its attention on as yet uncovered lines and branches. The resulting test suites significantly improve coverage over the current state of the art: compared to CodaMosa, a hybrid LLM / search-based software testing system, CoverUp substantially improves coverage across the board. On a per-module basis, CoverUp achieves median line coverage of 81% (vs. 62%), branch coverage of 53% (vs. 35%) and line+branch coverage of 78% (vs. 55%). We show that CoverUp's iterative, coverage-guided approach is crucial to its effectiveness, contributing to nearly half of its successes.
    

