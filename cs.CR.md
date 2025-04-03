# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance](https://arxiv.org/abs/2403.05842) | Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。 |

# 详细

[^1]: Hufu：一种通过置换等变性对预训练的Transformer进行水印处理的模态不可知水印系统

    Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance

    [https://arxiv.org/abs/2403.05842](https://arxiv.org/abs/2403.05842)

    Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。

    

    随着深度学习模型和服务的蓬勃发展，保护宝贵的模型参数免受盗窃已成为一项迫切关注的问题。水印技术被认为是所有权验证的重要工具。然而，当前的水印方案针对不同的模型和任务定制，难以作为集成的知识产权保护服务。我们提出了Hufu，这是一种针对预训练的基于Transformer的模型的模态不可知水印系统，依赖于Transformer的置换等变性质。Hufu通过微调预训练模型在特定置换的一组数据样本上嵌入水印，嵌入的模型基本上包含两组权重 -- 一组用于正常使用，另一组用于水印提取，触发条件是经过置换的输入。置换等变性确保这两组模型权重之间的最小干扰，从而在水印提取时具有高保真度。

    arXiv:2403.05842v1 Announce Type: cross  Abstract: With the blossom of deep learning models and services, it has become an imperative concern to safeguard the valuable model parameters from being stolen. Watermarking is considered an important tool for ownership verification. However, current watermarking schemes are customized for different models and tasks, hard to be integrated as an integrated intellectual protection service. We propose Hufu, a modality-agnostic watermarking system for pre-trained Transformer-based models, relying on the permutation equivariance property of Transformers. Hufu embeds watermark by fine-tuning the pre-trained model on a set of data samples specifically permuted, and the embedded model essentially contains two sets of weights -- one for normal use and the other for watermark extraction which is triggered on permuted inputs. The permutation equivariance ensures minimal interference between these two sets of model weights and thus high fidelity on downst
    

