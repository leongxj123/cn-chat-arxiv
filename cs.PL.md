# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatDBG: An AI-Powered Debugging Assistant](https://arxiv.org/abs/2403.16354) | ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。 |

# 详细

[^1]: ChatDBG: 一种基于人工智能的调试助手

    ChatDBG: An AI-Powered Debugging Assistant

    [https://arxiv.org/abs/2403.16354](https://arxiv.org/abs/2403.16354)

    ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。

    

    本文介绍了ChatDBG，这是第一个基于人工智能的调试助手。ChatDBG集成了大型语言模型(LLMs)，显著增强了传统调试器的功能和用户友好性。ChatDBG允许程序员与调试器进行协作对话，使他们能够提出关于程序状态的复杂问题，对崩溃或断言失败进行根本原因分析，并探索诸如“为什么x为空？”之类的开放性查询。为了处理这些查询，ChatDBG授予LLM自主权，通过发出命令来浏览堆栈和检查程序状态进行调试；然后报告其发现并将控制权交还给程序员。我们的ChatDBG原型与标准调试器集成，包括LLDB、GDB和WinDBG用于本地代码以及用于Python的Pdb。我们在各种代码集合上进行了评估，包括具有已知错误的C/C++代码和一套Python代码。

    arXiv:2403.16354v1 Announce Type: cross  Abstract: This paper presents ChatDBG, the first AI-powered debugging assistant. ChatDBG integrates large language models (LLMs) to significantly enhance the capabilities and user-friendliness of conventional debuggers. ChatDBG lets programmers engage in a collaborative dialogue with the debugger, allowing them to pose complex questions about program state, perform root cause analysis for crashes or assertion failures, and explore open-ended queries like "why is x null?". To handle these queries, ChatDBG grants the LLM autonomy to take the wheel and drive debugging by issuing commands to navigate through stacks and inspect program state; it then reports its findings and yields back control to the programmer. Our ChatDBG prototype integrates with standard debuggers including LLDB, GDB, and WinDBG for native code and Pdb for Python. Our evaluation across a diverse set of code, including C/C++ code with known bugs and a suite of Python code includi
    

