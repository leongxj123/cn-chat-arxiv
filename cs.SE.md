# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers.](http://arxiv.org/abs/2401.06461) | 本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。 |

# 详细

[^1]: 代码之间的界限：揭示机器和人类程序员之间不同的模式

    Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers. (arXiv:2401.06461v1 [cs.SE])

    [http://arxiv.org/abs/2401.06461](http://arxiv.org/abs/2401.06461)

    本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。

    

    大型语言模型在代码生成方面取得了显著的进展，但它们模糊了机器和人类源代码之间的区别，导致软件产物的完整性和真实性问题。本文通过对代码长度、词汇多样性和自然性等属性的严格分析，揭示了机器和人类代码固有的独特模式。在我们的研究中特别注意到，代码的结构分割是识别其来源的关键因素。基于我们的发现，我们提出了一种名为DetectCodeGPT的新型机器生成代码检测方法，该方法改进了DetectGPT。

    Large language models have catalyzed an unprecedented wave in code generation. While achieving significant advances, they blur the distinctions between machine-and human-authored source code, causing integrity and authenticity issues of software artifacts. Previous methods such as DetectGPT have proven effective in discerning machine-generated texts, but they do not identify and harness the unique patterns of machine-generated code. Thus, its applicability falters when applied to code. In this paper, we carefully study the specific patterns that characterize machine and human-authored code. Through a rigorous analysis of code attributes such as length, lexical diversity, and naturalness, we expose unique pat-terns inherent to each source. We particularly notice that the structural segmentation of code is a critical factor in identifying its provenance. Based on our findings, we propose a novel machine-generated code detection method called DetectCodeGPT, which improves DetectGPT by cap
    

