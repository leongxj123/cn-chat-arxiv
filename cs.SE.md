# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT.](http://arxiv.org/abs/2401.14279) | ZS4C提出了一种使用ChatGPT进行零射击合成可编译代码的轻量级方法，帮助用户重用或分析不完整的Q&A代码片段，通过识别缺失的导入语句并修复编译错误来实现。 |

# 详细

[^1]: ZS4C: 使用ChatGPT进行零射击合成不完整代码片段的可编译代码

    ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT. (arXiv:2401.14279v1 [cs.SE] CROSS LISTED)

    [http://arxiv.org/abs/2401.14279](http://arxiv.org/abs/2401.14279)

    ZS4C提出了一种使用ChatGPT进行零射击合成可编译代码的轻量级方法，帮助用户重用或分析不完整的Q&A代码片段，通过识别缺失的导入语句并修复编译错误来实现。

    

    技术问答（Q&A）网站如Stack Overflow已成为软件开发者寻求知识的重要来源。然而，Q&A网站上的代码片段通常由于未解析的类型和缺失的依赖库而无法编译和语义上不完整，这增加了用户重用或分析Q&A代码片段的障碍。之前的方法要么不适用于合成可编译代码，要么编译成功率低。为了解决这个问题，我们提出了ZS4C，一种使用大型语言模型（LLM）从不完整的代码片段中进行零射击合成可编译代码的轻量级方法。ZS4C分为两个阶段。在第一阶段，ZS4C利用一个LLM，即ChatGPT，根据我们设计的专用任务提示模板，为给定的代码片段识别缺失的导入语句。在第二阶段，ZS4C通过修复由于不正确的导入语句和语法错误引起的编译错误来修复代码。

    Technical question and answering (Q&A) sites such as Stack Overflow have become an important source for software developers to seek knowledge. However, code snippets on Q&A sites are usually uncompilable and semantically incomplete for compilation due to unresolved types and missing dependent libraries, which raises the obstacle for users to reuse or analyze Q&A code snippets. Prior approaches either are not designed for synthesizing compilable code or suffer from a low compilation success rate. To address this problem, we propose ZS4C, a lightweight approach to perform zero-shot synthesis of compilable code from incomplete code snippets using Large Language Model (LLM). ZS4C operates in two stages. In the first stage, ZS4C utilizes an LLM, i.e., ChatGPT, to identify missing import statements for a given code snippet, leveraging our designed task-specific prompt template. In the second stage, ZS4C fixes compilation errors caused by incorrect import statements and syntax errors through 
    

