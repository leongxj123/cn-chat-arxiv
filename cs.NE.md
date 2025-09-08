# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Human Curriculum Effects Emerge with In-Context Learning in Neural Networks](https://arxiv.org/abs/2402.08674) | 人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。 |

# 详细

[^1]: 使用上下文学习的神经网络中出现人类课程效应

    Human Curriculum Effects Emerge with In-Context Learning in Neural Networks

    [https://arxiv.org/abs/2402.08674](https://arxiv.org/abs/2402.08674)

    人类学习对示例课程和任务结构有敏感性。研究发现，在神经网络和语言模型中，通过上下文学习方法可以同时获得分组和交错训练的优势。

    

    人类学习对规则结构和训练中所使用的示例课程非常敏感。在由简洁规则控制的任务中，当相关示例在多次试验中被分组时，学习更加稳健；但在缺乏这样的规则的情况下，交错训练更加有效。迄今为止，没有神经模型能够同时捕捉到这些看似矛盾的效应。在本文中，我们展示了“上下文学习”（ICL）在使用元学习进行训练的神经网络和大型语言模型（LLMs）中自发产生了同样的权衡。ICL是通过内层循环算法在激活动力学中实现的一种“上下文内学习”（in-context learning）的能力，可以在没有权重更改的情况下学习新任务。对预训练的LLMs和元学习变压器进行的实验表明，ICL在涉及规则结构的任务中展示出了人类所示的分组优势，而同时进行权重学习则复制了人类在缺少这样结构的任务上所观察到的交错优势。

    Human learning is sensitive to rule-like structure and the curriculum of examples used for training. In tasks governed by succinct rules, learning is more robust when related examples are blocked across trials, but in the absence of such rules, interleaving is more effective. To date, no neural model has simultaneously captured these seemingly contradictory effects. Here we show that this same tradeoff spontaneously emerges with "in-context learning" (ICL) both in neural networks trained with metalearning and in large language models (LLMs). ICL is the ability to learn new tasks "in context" - without weight changes - via an inner-loop algorithm implemented in activation dynamics. Experiments with pretrained LLMs and metalearning transformers show that ICL exhibits the blocking advantage demonstrated in humans on a task involving rule-like structure, and conversely, that concurrent in-weight learning reproduces the interleaving advantage observed in humans on tasks lacking such structu
    

