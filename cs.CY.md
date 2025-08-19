# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://arxiv.org/abs/2403.20150) | TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。 |
| [^2] | [Large language models can replicate cross-cultural differences in personality.](http://arxiv.org/abs/2310.10679) | 大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。 |
| [^3] | [Unravelling Responsibility for AI.](http://arxiv.org/abs/2308.02608) | 本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。 |

# 详细

[^1]: TFB：面向时间序列预测方法全面且公平的基准比较

    TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

    [https://arxiv.org/abs/2403.20150](https://arxiv.org/abs/2403.20150)

    TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。

    

    时间序列会在经济、交通、健康和能源等不同领域中产生，对未来数值的预测在许多重要应用中起着关键作用。不出所料，许多预测方法被提出。为了确保进展，有必要能够以全面且可靠的方式经验性地研究和比较这些方法。为了实现这一目标，我们提出了TFB，一个自动化的时间序列预测（TSF）方法基准测试。TFB通过解决与数据集、比较方法和评估管道相关的缺点，推动了最新技术的发展：1）数据领域覆盖不足，2）对传统方法的刻板印象，3）不一致和不灵活的流程。为了获得更好的领域覆盖率，我们包括了来自10个不同领域的数据集：交通、电力、能源、环境、自然、经济、股票市场、银行、健康和网络。我们还提供了一个时间序列特性

    arXiv:2403.20150v1 Announce Type: cross  Abstract: Time series are generated in diverse domains such as economic, traffic, health, and energy, where forecasting of future values has numerous important applications. Not surprisingly, many forecasting methods are being proposed. To ensure progress, it is essential to be able to study and compare such methods empirically in a comprehensive and reliable manner. To achieve this, we propose TFB, an automated benchmark for Time Series Forecasting (TSF) methods. TFB advances the state-of-the-art by addressing shortcomings related to datasets, comparison methods, and evaluation pipelines: 1) insufficient coverage of data domains, 2) stereotype bias against traditional methods, and 3) inconsistent and inflexible pipelines. To achieve better domain coverage, we include datasets from 10 different domains: traffic, electricity, energy, the environment, nature, economic, stock markets, banking, health, and the web. We also provide a time series char
    
[^2]: 大型语言模型可以复制跨文化个性差异

    Large language models can replicate cross-cultural differences in personality. (arXiv:2310.10679v1 [cs.CL])

    [http://arxiv.org/abs/2310.10679](http://arxiv.org/abs/2310.10679)

    大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。

    

    我们使用一项大规模实验(N=8000)来确定GPT-4是否可以复制使用十项人格问卷测量的大五人格的跨文化差异。我们选择美国和韩国作为文化对比，因为先前的研究表明这两个国家的人之间存在显著的人格差异。我们操纵了模拟的目标（美国 vs. 韩国），问卷的语言（英语 vs. 韩语）以及语言模型（GPT-4 vs. GPT-3.5）。我们的结果表明，GPT-4复制了每个因子的跨文化差异。然而，平均评级具有上升偏差，并且比人类样本的变异性更低，以及结构效度较低。总的来说，我们提供了初步的证据说明LLMs可以促进跨文化心理研究。

    We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. Overall, we provide preliminary evidence that LLMs can aid cross-cultural psychological research.
    
[^3]: 解构人工智能责任

    Unravelling Responsibility for AI. (arXiv:2308.02608v1 [cs.AI])

    [http://arxiv.org/abs/2308.02608](http://arxiv.org/abs/2308.02608)

    本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。

    

    为了在涉及人工智能系统的复杂情况下合理思考责任应该放在何处，我们首先需要一个足够清晰和详细的跨学科词汇来谈论责任。责任是一种三元关系，涉及到一个行为者、一个事件和一种责任方式。作为一种有意识的为了支持对人工智能责任进行实践推理的“解构”责任概念的努力，本文采取了“行为者A对事件O负责”的三部分表述，并确定了A、负责、O的子类别的有效组合。这些有效组合我们称之为“责任串”，分为四种责任意义：角色责任、因果责任、法律责任和道德责任。我们通过两个运行示例进行了说明，一个涉及医疗AI系统，另一个涉及AV与行人的致命碰撞。

    To reason about where responsibility does and should lie in complex situations involving AI-enabled systems, we first need a sufficiently clear and detailed cross-disciplinary vocabulary for talking about responsibility. Responsibility is a triadic relation involving an actor, an occurrence, and a way of being responsible. As part of a conscious effort towards 'unravelling' the concept of responsibility to support practical reasoning about responsibility for AI, this paper takes the three-part formulation, 'Actor A is responsible for Occurrence O' and identifies valid combinations of subcategories of A, is responsible for, and O. These valid combinations - which we term "responsibility strings" - are grouped into four senses of responsibility: role-responsibility; causal responsibility; legal liability-responsibility; and moral responsibility. They are illustrated with two running examples, one involving a healthcare AI-based system and another the fatal collision of an AV with a pedes
    

