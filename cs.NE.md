# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recurrent Transformers with Dynamic Halt](https://rss.arxiv.org/abs/2402.00976) | 本文研究了增强Transformer与循环机制的两种方法，并提出了新的扩展和组合方法。在多个诊断任务中进行比较，探索它们的归纳偏好。 |

# 详细

[^1]: 具有动态停止的循环Transformer

    Recurrent Transformers with Dynamic Halt

    [https://rss.arxiv.org/abs/2402.00976](https://rss.arxiv.org/abs/2402.00976)

    本文研究了增强Transformer与循环机制的两种方法，并提出了新的扩展和组合方法。在多个诊断任务中进行比较，探索它们的归纳偏好。

    

    本文研究了两种主要方法在增强Transformer与循环机制方面的归纳偏好——（1）类似于Universal Transformers的深度逐层循环方法；和（2）类似于Temporal Latent Bottleneck的分块时态循环方法。此外，我们提出并研究了扩展和组合上述方法的新方式，例如，我们提出了一种基于全局均值的Universal Transformer动态停止机制，并将Universal Transformer的元素融入到Temporal Latent Bottleneck中。我们通过多个诊断任务（如Long Range Arena（LRA），翻转-翻转语言建模，ListOps和逻辑推理）比较了模型并探索了它们的归纳偏好。

    In this paper, we study the inductive biases of two major approaches to augmenting Transformers with a recurrent mechanism - (1) the approach of incorporating a depth-wise recurrence similar to Universal Transformers; and (2) the approach of incorporating a chunk-wise temporal recurrence like Temporal Latent Bottleneck. Furthermore, we propose and investigate novel ways to extend and combine the above methods - for example, we propose a global mean-based dynamic halting mechanism for Universal Transformer and an augmentation of Temporal Latent Bottleneck with elements from Universal Transformer. We compare the models and probe their inductive biases in several diagnostic tasks such as Long Range Arena (LRA), flip-flop language modeling, ListOps, and Logical Inference.
    

