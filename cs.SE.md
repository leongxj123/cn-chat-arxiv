# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving LLM Code Generation with Grammar Augmentation](https://arxiv.org/abs/2403.01632) | SynCode是一个新框架，结合编程语言的语法和DFA mask store，在LLMs中生成代码过程中获得96.07%的句法错误降低，并展现出提高句法精度的重大影响。 |

# 详细

[^1]: 通过语法增强改进LLM代码生成

    Improving LLM Code Generation with Grammar Augmentation

    [https://arxiv.org/abs/2403.01632](https://arxiv.org/abs/2403.01632)

    SynCode是一个新框架，结合编程语言的语法和DFA mask store，在LLMs中生成代码过程中获得96.07%的句法错误降低，并展现出提高句法精度的重大影响。

    

    我们提出了 SynCode，一个用于高效和通用地解码大型语言模型（LLMs）代码的新框架。SynCode利用编程语言的语法，利用离线构建的基于语言语法终结符的高效查找表DFA mask store。我们展示了SynCode在给定编程语言的上下文无关文法（CFG）的完备性和正确性，展示其在保留语义上有效令牌的同时拒绝无效令牌的能力。该框架与由CFG定义的任何语言无缝集成，验证了针对Python和Go的CFG实验。结果突出了当SynCode与最先进的LLMs结合时，语法错误减少96.07%，彰显了其对提高代码生成中的句法精度的重大影响。

    arXiv:2403.01632v1 Announce Type: new  Abstract: We present SynCode a novel framework for efficient and general syntactical decoding of code with large language models (LLMs). SynCode leverages the grammar of a programming language, utilizing an offline-constructed efficient lookup table called DFA mask store based on language grammar terminals. We demonstrate SynCode's soundness and completeness given the context-free grammar (CFG) of the programming language, presenting its ability to retain syntactically valid tokens while rejecting invalid ones. The framework seamlessly integrates with any language defined by CFG, as evidenced by experiments on CFGs for Python and Go. The results underscore the significant reduction of 96.07% of syntax errors achieved when SynCode is combined with state-of-the-art LLMs, showcasing its substantial impact on enhancing syntactical precision in code generation.   Our code is available at https://github.com/uiuc-focal-lab/syncode.
    

