# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Few-Shot Cross-System Anomaly Trace Classification for Microservice-based systems](https://arxiv.org/abs/2403.18998) | 提出了针对微服务系统的少样本异常跟踪分类的新框架，利用多头注意力自编码器构建系统特定的跟踪表示，并应用基于Transformer编码器的模型无关元学习进行高效分类。 |
| [^2] | [ChatDBG: An AI-Powered Debugging Assistant](https://arxiv.org/abs/2403.16354) | ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。 |

# 详细

[^1]: 微服务系统的少样本跨系统异常跟踪分类

    Few-Shot Cross-System Anomaly Trace Classification for Microservice-based systems

    [https://arxiv.org/abs/2403.18998](https://arxiv.org/abs/2403.18998)

    提出了针对微服务系统的少样本异常跟踪分类的新框架，利用多头注意力自编码器构建系统特定的跟踪表示，并应用基于Transformer编码器的模型无关元学习进行高效分类。

    

    微服务系统（MSS）由于其复杂和动态的特性可能在各种故障类别中出现故障。为了有效处理故障，AIOps工具利用基于跟踪的异常检测和根本原因分析。本文提出了一个新颖的框架，用于微服务系统的少样本异常跟踪分类。我们的框架包括两个主要组成部分：（1）多头注意力自编码器用于构建系统特定的跟踪表示，从而实现（2）基于Transformer编码器的模型无关元学习，以进行有效和高效的少样本异常跟踪分类。该框架在两个代表性的MSS，Trainticket和OnlineBoutique上进行了评估，使用开放数据集。结果表明，我们的框架能够调整学到的知识，以对新的、未见的新颖故障类别的异常跟踪进行分类，无论是在最初训练的同一系统内，还是在其他系统中。

    arXiv:2403.18998v1 Announce Type: cross  Abstract: Microservice-based systems (MSS) may experience failures in various fault categories due to their complex and dynamic nature. To effectively handle failures, AIOps tools utilize trace-based anomaly detection and root cause analysis. In this paper, we propose a novel framework for few-shot abnormal trace classification for MSS. Our framework comprises two main components: (1) Multi-Head Attention Autoencoder for constructing system-specific trace representations, which enables (2) Transformer Encoder-based Model-Agnostic Meta-Learning to perform effective and efficient few-shot learning for abnormal trace classification. The proposed framework is evaluated on two representative MSS, Trainticket and OnlineBoutique, with open datasets. The results show that our framework can adapt the learned knowledge to classify new, unseen abnormal traces of novel fault categories both within the same system it was initially trained on and even in the 
    
[^2]: ChatDBG: 一种基于人工智能的调试助手

    ChatDBG: An AI-Powered Debugging Assistant

    [https://arxiv.org/abs/2403.16354](https://arxiv.org/abs/2403.16354)

    ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。

    

    本文介绍了ChatDBG，这是第一个基于人工智能的调试助手。ChatDBG集成了大型语言模型(LLMs)，显著增强了传统调试器的功能和用户友好性。ChatDBG允许程序员与调试器进行协作对话，使他们能够提出关于程序状态的复杂问题，对崩溃或断言失败进行根本原因分析，并探索诸如“为什么x为空？”之类的开放性查询。为了处理这些查询，ChatDBG授予LLM自主权，通过发出命令来浏览堆栈和检查程序状态进行调试；然后报告其发现并将控制权交还给程序员。我们的ChatDBG原型与标准调试器集成，包括LLDB、GDB和WinDBG用于本地代码以及用于Python的Pdb。我们在各种代码集合上进行了评估，包括具有已知错误的C/C++代码和一套Python代码。

    arXiv:2403.16354v1 Announce Type: cross  Abstract: This paper presents ChatDBG, the first AI-powered debugging assistant. ChatDBG integrates large language models (LLMs) to significantly enhance the capabilities and user-friendliness of conventional debuggers. ChatDBG lets programmers engage in a collaborative dialogue with the debugger, allowing them to pose complex questions about program state, perform root cause analysis for crashes or assertion failures, and explore open-ended queries like "why is x null?". To handle these queries, ChatDBG grants the LLM autonomy to take the wheel and drive debugging by issuing commands to navigate through stacks and inspect program state; it then reports its findings and yields back control to the programmer. Our ChatDBG prototype integrates with standard debuggers including LLDB, GDB, and WinDBG for native code and Pdb for Python. Our evaluation across a diverse set of code, including C/C++ code with known bugs and a suite of Python code includi
    

