# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CodeMind: A Framework to Challenge Large Language Models for Code Reasoning](https://arxiv.org/abs/2402.09664) | CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。 |

# 详细

[^1]: CodeMind:一个用于挑战大型语言模型进行代码推理的框架

    CodeMind: A Framework to Challenge Large Language Models for Code Reasoning

    [https://arxiv.org/abs/2402.09664](https://arxiv.org/abs/2402.09664)

    CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    

    仅靠测试通过来评估大型语言模型（LLMs）的代码合成能力可能会导致不公正的评估或促进具有数据泄漏的模型，作为一种替代方案，我们介绍了CodeMind，这是一个旨在评估LLMs的代码推理能力的框架。CodeMind目前支持三种代码推理任务：独立执行推理（IER）、依赖执行推理（DER）和规范推理（SR）。前两者评估模型以预测任意代码的执行输出，或者模型能够正确合成的代码。第三个任务评估LLMs实现指定预期行为的程度。我们使用CodeMind对两种不同编程语言中的五个基准下的九个LLMs进行了广泛的评估，结果表明LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    arXiv:2402.09664v1 Announce Type: cross  Abstract: Solely relying on test passing to evaluate Large Language Models (LLMs) for code synthesis may result in unfair assessment or promoting models with data leakage. As an alternative, we introduce CodeMind, a framework designed to gauge the code reasoning abilities of LLMs. CodeMind currently supports three code reasoning tasks: Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR). The first two evaluate models to predict the execution output of an arbitrary code or code the model could correctly synthesize. The third one evaluates the extent to which LLMs implement the specified expected behavior. Our extensive evaluation of nine LLMs across five benchmarks in two different programming languages using CodeMind shows that LLMs fairly understand control flow constructs and, in general, are capable of reasoning how inputs evolve to output, specifically for simple programs and the ones 
    

