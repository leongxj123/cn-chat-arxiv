# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structured Tree Alignment for Evaluation of (Speech) Constituency Parsing](https://arxiv.org/abs/2402.13433) | 提出了一种受语音解析器评估问题启发的结构化句法分析树相似性度量指标STRUCT-IOU，有效地比较了口语词边界上的组块分析树与书面词上基准解析之间的差异，并展示了在文本组块分析评估中的优越性。 |

# 详细

[^1]: 结构化树对齐用于（语音）组块分析评估

    Structured Tree Alignment for Evaluation of (Speech) Constituency Parsing

    [https://arxiv.org/abs/2402.13433](https://arxiv.org/abs/2402.13433)

    提出了一种受语音解析器评估问题启发的结构化句法分析树相似性度量指标STRUCT-IOU，有效地比较了口语词边界上的组块分析树与书面词上基准解析之间的差异，并展示了在文本组块分析评估中的优越性。

    

    我们提出了结构化平均交集-联盟比（STRUCT-IOU），这是一种句法分析树之间的相似性度量指标，受到了评估语音解析器问题的启发。STRUCT-IOU使得可以比较在自动识别的口语词边界上的组块分析树与基准解析（在书面词上）之间的差异。为了计算这个指标，我们通过强制对齐将基准解析树投影到语音领域，将投影的基准成分与预测的成分在一定的结构约束下对齐，然后计算所有对齐成分对之间的平均IOU分数。STRUCT-IOU考虑了词边界，并克服了预测的词和基准事实可能没有完美一一对应的挑战。扩展到文本组块分析的评估，我们展示STRUCT-IOU表现出更高的对句法合理解析的容忍度。

    arXiv:2402.13433v1 Announce Type: new  Abstract: We present the structured average intersection-over-union ratio (STRUCT-IOU), a similarity metric between constituency parse trees motivated by the problem of evaluating speech parsers. STRUCT-IOU enables comparison between a constituency parse tree (over automatically recognized spoken word boundaries) with the ground-truth parse (over written words). To compute the metric, we project the ground-truth parse tree to the speech domain by forced alignment, align the projected ground-truth constituents with the predicted ones under certain structured constraints, and calculate the average IOU score across all aligned constituent pairs. STRUCT-IOU takes word boundaries into account and overcomes the challenge that the predicted words and ground truth may not have perfect one-to-one correspondence. Extending to the evaluation of text constituency parsing, we demonstrate that STRUCT-IOU shows higher tolerance to syntactically plausible parses 
    

