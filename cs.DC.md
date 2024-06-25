# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OMPGPT: A Generative Pre-trained Transformer Model for OpenMP.](http://arxiv.org/abs/2401.16445) | OMPGPT是一种为了OpenMP pragma生成而设计的生成式预训练Transformer模型，采用了来自NLP领域的提示工程技术，并创建了一种创新的策略chain-of-OMP。 |

# 详细

[^1]: OMPGPT: 一种用于OpenMP的生成式预训练Transformer模型

    OMPGPT: A Generative Pre-trained Transformer Model for OpenMP. (arXiv:2401.16445v1 [cs.SE])

    [http://arxiv.org/abs/2401.16445](http://arxiv.org/abs/2401.16445)

    OMPGPT是一种为了OpenMP pragma生成而设计的生成式预训练Transformer模型，采用了来自NLP领域的提示工程技术，并创建了一种创新的策略chain-of-OMP。

    

    大型语言模型（LLMs），如ChatGPT等模型，已经在自然语言处理领域引起了革命。随着这一趋势，基于代码的大型语言模型，如StarCoder、WizardCoder和CodeLlama等，已经涌现出来，在大量的代码数据库上进行了广泛的训练。然而，由于设计固有的原因，这些模型主要关注代码生成、代码完成和注释生成等生成任务，以及对多种编程语言的一般支持。虽然代码LLMs的通用能力对许多程序员来说很有用，但高性能计算（HPC）领域具有更窄的需求集，使得更小、更具领域特定的LM成为一个更明智的选择。本文介绍了OMPGPT，这是一种精心设计的新型模型，旨在充分利用语言模型在OpenMP pragma生成方面的固有优势。此外，我们采用并改进了来自NLP领域的提示工程技术，创建了链式OMP（chain-of-OMP），这是一种创新策略。

    Large language models (LLMs), as epitomized by models like ChatGPT, have revolutionized the field of natural language processing (NLP). Along with this trend, code-based large language models such as StarCoder, WizardCoder, and CodeLlama have emerged, trained extensively on vast repositories of code data. Yet, inherent in their design, these models primarily focus on generative tasks like code generation, code completion, and comment generation, and general support for multiple programming languages. While the generic abilities of code LLMs are useful for many programmers, the area of high-performance computing (HPC) has a narrower set of requirements that make a smaller and more domain-specific LM a smarter choice. This paper introduces OMPGPT, a novel model meticulously designed to harness the inherent strengths of language models for OpenMP pragma generation. Furthermore, we adopt and adapt prompt engineering techniques from the NLP domain to create chain-of-OMP, an innovative strat
    

