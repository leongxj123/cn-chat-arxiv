# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models.](http://arxiv.org/abs/2301.06267) | 通过跨模态适应方法，在多模态模型下利用少样本示例（包括文本和声音）进行狗的视觉分类，并取得了最先进的结果。 |

# 详细

[^1]: 多模态有助于单模态：多模态模型下的交叉模态少样本学习

    Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models. (arXiv:2301.06267v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2301.06267](http://arxiv.org/abs/2301.06267)

    通过跨模态适应方法，在多模态模型下利用少样本示例（包括文本和声音）进行狗的视觉分类，并取得了最先进的结果。

    

    快速学习新任务的能力是智能代理的核心要素，也被称为少样本学习。传统的少样本学习基准使用来自单模态的少样本样本，但这些样本可能不足以描述整个概念类。相比之下，人类使用跨模态信息高效地学习新概念。在这项工作中，我们展示了通过阅读关于狗并听它们吠叫的声音来构建更好的视觉狗分类器的可能性。为此，我们利用最近的多模态基础模型（如CLIP）是固有的跨模态的特性，将不同的模态映射到相同的表示空间。具体而言，我们提出了一种简单的跨模态适应方法，从跨越不同模态的少样本示例中进行学习。通过将类名重新用作额外的一次性训练样本，我们使用一个极其简单的线性分类器实现了最先进的结果。

    The ability to quickly learn a new task with minimal instruction - known as few-shot learning - is a central aspect of intelligent agents. Classical few-shot benchmarks make use of few-shot samples from a single modality, but such samples may not be sufficient to characterize an entire concept class. In contrast, humans use cross-modal information to learn new concepts efficiently. In this work, we demonstrate that one can indeed build a better ${\bf visual}$ dog classifier by ${\bf read}$ing about dogs and ${\bf listen}$ing to them bark. To do so, we exploit the fact that recent multimodal foundation models such as CLIP are inherently cross-modal, mapping different modalities to the same representation space. Specifically, we propose a simple cross-modal adaptation approach that learns from few-shot examples spanning different modalities. By repurposing class names as additional one-shot training samples, we achieve SOTA results with an embarrassingly simple linear classifier for visi
    

