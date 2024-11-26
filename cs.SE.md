# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When Dataflow Analysis Meets Large Language Models](https://arxiv.org/abs/2402.10754) | 这个研究提出了一个由大型语言模型驱动的数据流分析框架，可以分析任意代码片段，无需编译基础设施，并自动合成下游应用，有效解决数据流相关漏洞检测问题 |
| [^2] | [SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics.](http://arxiv.org/abs/2401.09622) | SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。 |

# 详细

[^1]: 当数据流分析遇上大型语言模型

    When Dataflow Analysis Meets Large Language Models

    [https://arxiv.org/abs/2402.10754](https://arxiv.org/abs/2402.10754)

    这个研究提出了一个由大型语言模型驱动的数据流分析框架，可以分析任意代码片段，无需编译基础设施，并自动合成下游应用，有效解决数据流相关漏洞检测问题

    

    数据流分析是一种强大的代码分析技术，可以推断程序值之间的依赖关系，支持代码优化、程序理解和错误检测。本文介绍了LLMDFA，这是一个由LLM驱动的数据流分析框架，可以分析任意代码片段，无需编译基础设施，并自动合成下游应用。LLMDFA受基于摘要的数据流分析启发，将问题分解为三个子问题，并通过几种关键策略有效解决，包括少样本链式思维提示和工具合成。我们的评估表明，该设计可以减轻幻觉并提高推理能力，在检测基准测试中获取高精度和召回率

    arXiv:2402.10754v1 Announce Type: cross  Abstract: Dataflow analysis is a powerful code analysis technique that reasons dependencies between program values, offering support for code optimization, program comprehension, and bug detection. Existing approaches require the successful compilation of the subject program and customizations for downstream applications. This paper introduces LLMDFA, an LLM-powered dataflow analysis framework that analyzes arbitrary code snippets without requiring a compilation infrastructure and automatically synthesizes downstream applications. Inspired by summary-based dataflow analysis, LLMDFA decomposes the problem into three sub-problems, which are effectively resolved by several essential strategies, including few-shot chain-of-thought prompting and tool synthesis. Our evaluation has shown that the design can mitigate the hallucination and improve the reasoning ability, obtaining high precision and recall in detecting dataflow-related bugs upon benchmark
    
[^2]: SMOOTHIE: 软件分析的超参数优化理论

    SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics. (arXiv:2401.09622v1 [cs.SE])

    [http://arxiv.org/abs/2401.09622](http://arxiv.org/abs/2401.09622)

    SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。

    

    超参数优化是调整学习器控制参数的黑魔法。在软件分析中，经常发现调优可以带来显著的性能改进。尽管如此，超参数优化在软件分析中通常被很少或很差地应用，可能是因为探索所有参数选项的CPU成本太高。我们假设当损失函数的“光滑度”更好时，学习器的泛化能力更强。这个理论非常有用，因为可以很快测试不同超参数选择对“光滑度”的影响（例如，对于深度学习器，在一个epoch之后就可以进行测试）。为了测试这个理论，本文实现和测试了SMOOTHIE，一种通过考虑“光滑度”来引导优化的新型超参数优化器。本文的实验将SMOOTHIE应用于多个软件工程任务，包括（a）GitHub问题寿命预测；（b）静态代码警告中错误警报的检测；（c）缺陷预测。

    Hyper-parameter optimization is the black art of tuning a learner's control parameters. In software analytics, a repeated result is that such tuning can result in dramatic performance improvements. Despite this, hyper-parameter optimization is often applied rarely or poorly in software analytics--perhaps due to the CPU cost of exploring all those parameter options can be prohibitive.  We theorize that learners generalize better when the loss landscape is ``smooth''. This theory is useful since the influence on ``smoothness'' of different hyper-parameter choices can be tested very quickly (e.g. for a deep learner, after just one epoch).  To test this theory, this paper implements and tests SMOOTHIE, a novel hyper-parameter optimizer that guides its optimizations via considerations of ``smothness''. The experiments of this paper test SMOOTHIE on numerous SE tasks including (a) GitHub issue lifetime prediction; (b) detecting false alarms in static code warnings; (c) defect prediction, and
    

