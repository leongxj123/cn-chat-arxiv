# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision](https://arxiv.org/abs/2311.02333) | 本文提出了一种基于字节级精度的编码器-解码器模型，用于理解DNA的自然语言。该模型可以在字节级精度上分析DNA序列，使得能够用于识别DNA序列中的各种功能和变异。 |

# 详细

[^1]: 使用字节级精确度的编码器-解码器基于Transformer模型理解DNA的自然语言

    Understanding the Natural Language of DNA using Encoder-Decoder Foundation Models with Byte-level Precision

    [https://arxiv.org/abs/2311.02333](https://arxiv.org/abs/2311.02333)

    本文提出了一种基于字节级精度的编码器-解码器模型，用于理解DNA的自然语言。该模型可以在字节级精度上分析DNA序列，使得能够用于识别DNA序列中的各种功能和变异。

    

    本文提出了一种基于字节级编码器-解码器Transformer架构的集合核苷酸字节级编码器-解码器(ENBED)基础模型，可以在字节级精度上分析DNA序列。ENBED使用次二次的注意力实现了一个高效的模型，能够进行序列到序列的转换，泛化先前基因组模型只采用编码器或者解码器体系结构的限制。我们使用遮蔽语言建模来预训练这个基础模型，使用参考基因组序列并将其应用到以下下游任务上：(1)识别增强子、启动子和剪切位点，(2)识别包含碱基调用不匹配和插入/缺失错误的序列，这是对多个碱基对进行标记化的方案的优势，丢失了字节级精度分析的能力，(3)识别基因组序列的生物功能注释，以及(4)生成突变基因组序列。

    arXiv:2311.02333v2 Announce Type: replace Abstract: This paper presents the Ensemble Nucleotide Byte-level Encoder-Decoder (ENBED) foundation model, analyzing DNA sequences at byte-level precision with an encoder-decoder Transformer architecture. ENBED uses a sub-quadratic implementation of attention to develop an efficient model capable of sequence-to-sequence transformations, generalizing previous genomic models with encoder-only or decoder-only architectures. We use Masked Language Modeling to pre-train the foundation model using reference genome sequences and apply it in the following downstream tasks: (1) identification of enhancers, promotors and splice sites, (2) recognition of sequences containing base call mismatches and insertion/deletion errors, an advantage over tokenization schemes involving multiple base pairs, which lose the ability to analyze with byte-level precision, (3) identification of biological function annotations of genomic sequences, and (4) generating mutatio
    

