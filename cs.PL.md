# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Constrained Decoding for Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars](https://arxiv.org/abs/2402.17988) | 本文提出了一种增量解析器，通过对上下文敏感语法进行高效左右商，实现了对语法正确性的早期拒绝和对完整程序的有效检测。 |

# 详细

[^1]: 通过对上下文敏感语法进行高效左右商，在代码语言模型的约束解码

    Constrained Decoding for Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars

    [https://arxiv.org/abs/2402.17988](https://arxiv.org/abs/2402.17988)

    本文提出了一种增量解析器，通过对上下文敏感语法进行高效左右商，实现了对语法正确性的早期拒绝和对完整程序的有效检测。

    

    大型语言模型是程序合成和高级自动完成的强大工具，但不能保证其输出代码在语法上是正确的。本文提出了一种增量解析器，允许早期拒绝语法上不正确的代码，并且能够有效检测用于填充任务的完整程序。我们开发了能够在任意上下文无关语法的左右商上操作的Earley式解析器，并将增量解析和商操作扩展到许多常见编程语言的语法中存在的几个上下文敏感特性。这些贡献的结果是一种高效、通用和扎实的左右商解析方法。

    arXiv:2402.17988v1 Announce Type: cross  Abstract: Large Language Models are powerful tools for program synthesis and advanced auto-completion, but come with no guarantee that their output code is syntactically correct. This paper contributes an incremental parser that allows early rejection of syntactically incorrect code, as well as efficient detection of complete programs for fill-in-the-middle (FItM) tasks. We develop Earley-style parsers that operate over left and right quotients of arbitrary context-free grammars, and we extend our incremental parsing and quotient operations to several context-sensitive features present in the grammars of many common programming languages. The result of these contributions is an efficient, general, and well-grounded method for left and right quotient parsing.   To validate our theoretical contributions -- and the practical effectiveness of certain design decisions -- we evaluate our method on the particularly difficult case of FItM completion for
    

