# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Activation Steering for Robust Type Prediction in CodeLLMs](https://arxiv.org/abs/2404.01903) | 我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。 |

# 详细

[^1]: 在CodeLLMs中实现类型预测的鲁棒激活导向技术

    Activation Steering for Robust Type Prediction in CodeLLMs

    [https://arxiv.org/abs/2404.01903](https://arxiv.org/abs/2404.01903)

    我们提出了一种激活导向技术，通过编辑模型内部激活来改善CodeLLMs在代码类型预测中对于语法干扰的鲁棒性，并成功应用于Python和TypeScript的类型预测，将类型误差率纠正高达90%。

    

    预训练在代码上的现代LLMs能够成功地完成各种编程任务。然而，它们的性能对语法特征非常敏感，例如变量和类型的名称、代码结构以及类型提示的存在。我们提出了一种推理时技术，使CodeLLMs更能抵御语法干扰因素，这些因素与语义无关。我们的方法依赖于激活导向，涉及编辑内部模型激活以将模型引导到正确的预测。我们通过从突变测试中汲取灵感构建激活向量的新方法，该方法构建最小的破坏语义的代码编辑。相比之下，我们从保留语义的代码编辑中构建激活向量。我们将我们的方法应用于逐渐类型化语言Python和TypeScript的类型预测任务。这种方法可以纠正高达90%的类型错误预测。

    arXiv:2404.01903v1 Announce Type: new  Abstract: Contemporary LLMs pretrained on code are capable of succeeding at a wide variety of programming tasks. However, their performance is very sensitive to syntactic features, such as the names of variables and types, the structure of code, and presence of type hints. We contribute an inference-time technique to make CodeLLMs more robust to syntactic distractors that are semantically irrelevant. Our methodology relies on activation steering, which involves editing internal model activations to steer the model towards the correct prediction. We contribute a novel way to construct steering vectors by taking inspiration from mutation testing, which constructs minimal semantics-breaking code edits. In contrast, we construct steering vectors from semantics-preserving code edits. We apply our approach to the task of type prediction for the gradually typed languages Python and TypeScript. This approach corrects up to 90% of type mispredictions. Fina
    

