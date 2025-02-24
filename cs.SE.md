# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models.](http://arxiv.org/abs/2306.03268) | 本文主张在大型预训练模型的潮流中，还应推广面向特定领域的预训练模型，并以 StackOverflow 为例展示了其优越性。 |

# 详细

[^1]: 面向特定领域的预训练模型：相比一锅粥式模型，千万不要让领域的供给不足受到波及

    Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models. (arXiv:2306.03268v1 [cs.CL])

    [http://arxiv.org/abs/2306.03268](http://arxiv.org/abs/2306.03268)

    本文主张在大型预训练模型的潮流中，还应推广面向特定领域的预训练模型，并以 StackOverflow 为例展示了其优越性。

    

    大型预训练神经语言模型（如OpenAI的GPT系列）为NLP和软件工程带来了极大的进展。然而，我们认为这种追求大而全的潮流应该与针对特定目的、规模适中的预训练模型相结合。本文以StackOverflow为例，展示了我们的面向特定领域的预训练模型相对于通用模型在验证困惑度和迁移学习准确性方面表现更优。

    Large pre-trained neural language models have brought immense progress to both NLP and software engineering. Models in OpenAI's GPT series now dwarf Google's BERT and Meta's RoBERTa, which previously set new benchmarks on a wide range of NLP applications. These models are trained on massive corpora of heterogeneous data from web crawls, which enables them to learn general language patterns and semantic relationships. However, the largest models are both expensive to train and deploy and are often closed-source, so we lack access to their data and design decisions. We argue that this trend towards large, general-purpose models should be complemented with single-purpose, more modestly sized pre-trained models. In this work, we take StackOverflow (SO) as a domain example in which large volumes of rich aligned code and text data is available. We adopt standard practices for pre-training large language models, including using a very large context size (2,048 tokens), batch size (0.5M tokens
    

