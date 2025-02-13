# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do Large Code Models Understand Programming Concepts? A Black-box Approach](https://arxiv.org/abs/2402.05980) | 本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。 |

# 详细

[^1]: 大型代码模型是否理解编程概念？一种黑盒方法探究

    Do Large Code Models Understand Programming Concepts? A Black-box Approach

    [https://arxiv.org/abs/2402.05980](https://arxiv.org/abs/2402.05980)

    本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。

    

    大型语言模型在文本生成方面的成功也使其在代码生成和编码任务方面表现更好。虽然有很多工作展示了它们在代码补全和编辑等任务上的出色性能，但为什么它们能够成功还不清楚。我们通过探索自回归模型对底层程序的逻辑结构理解程度，来填补这一差距。我们提出了用于编程概念谓词的反事实分析（CACP）作为一种反事实测试框架，以评估大型代码模型是否理解编程概念。只通过黑盒访问模型，我们使用CACP评估了十个流行的大型代码模型对四个不同编程概念的理解情况。我们的研究结果表明，当前模型缺乏对数据流和控制流等概念的理解。

    Large Language Models' success on text generation has also made them better at code generation and coding tasks. While a lot of work has demonstrated their remarkable performance on tasks such as code completion and editing, it is still unclear as to why. We help bridge this gap by exploring to what degree auto-regressive models understand the logical constructs of the underlying programs. We propose Counterfactual Analysis for Programming Concept Predicates (CACP) as a counterfactual testing framework to evaluate whether Large Code Models understand programming concepts. With only black-box access to the model, we use CACP to evaluate ten popular Large Code Models for four different programming concepts. Our findings suggest that current models lack understanding of concepts such as data flow and control flow.
    

