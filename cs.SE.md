# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics.](http://arxiv.org/abs/2401.09622) | SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。 |

# 详细

[^1]: SMOOTHIE: 软件分析的超参数优化理论

    SMOOTHIE: A Theory of Hyper-parameter Optimization for Software Analytics. (arXiv:2401.09622v1 [cs.SE])

    [http://arxiv.org/abs/2401.09622](http://arxiv.org/abs/2401.09622)

    SMOOTHIE是一种通过考虑损失函数的“光滑度”来引导超参数优化的新型方法，在软件分析中应用可以带来显著的性能改进。

    

    超参数优化是调整学习器控制参数的黑魔法。在软件分析中，经常发现调优可以带来显著的性能改进。尽管如此，超参数优化在软件分析中通常被很少或很差地应用，可能是因为探索所有参数选项的CPU成本太高。我们假设当损失函数的“光滑度”更好时，学习器的泛化能力更强。这个理论非常有用，因为可以很快测试不同超参数选择对“光滑度”的影响（例如，对于深度学习器，在一个epoch之后就可以进行测试）。为了测试这个理论，本文实现和测试了SMOOTHIE，一种通过考虑“光滑度”来引导优化的新型超参数优化器。本文的实验将SMOOTHIE应用于多个软件工程任务，包括（a）GitHub问题寿命预测；（b）静态代码警告中错误警报的检测；（c）缺陷预测。

    Hyper-parameter optimization is the black art of tuning a learner's control parameters. In software analytics, a repeated result is that such tuning can result in dramatic performance improvements. Despite this, hyper-parameter optimization is often applied rarely or poorly in software analytics--perhaps due to the CPU cost of exploring all those parameter options can be prohibitive.  We theorize that learners generalize better when the loss landscape is ``smooth''. This theory is useful since the influence on ``smoothness'' of different hyper-parameter choices can be tested very quickly (e.g. for a deep learner, after just one epoch).  To test this theory, this paper implements and tests SMOOTHIE, a novel hyper-parameter optimizer that guides its optimizations via considerations of ``smothness''. The experiments of this paper test SMOOTHIE on numerous SE tasks including (a) GitHub issue lifetime prediction; (b) detecting false alarms in static code warnings; (c) defect prediction, and
    

