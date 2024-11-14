# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Program Repair with Semantic-Preserving Transformations: A Naturalness Assessment](https://arxiv.org/abs/2402.11892) | 本文研究了保留语义的转换的自然性及其对NPR评估的影响，发现了NPR系统在面对不自然的代码转换时会产生较高的误报率，且在使用自然转换进行评估时性能明显下降。 |

# 详细

[^1]: 用保留语义的转换评估程序修复：自然性评估

    Evaluating Program Repair with Semantic-Preserving Transformations: A Naturalness Assessment

    [https://arxiv.org/abs/2402.11892](https://arxiv.org/abs/2402.11892)

    本文研究了保留语义的转换的自然性及其对NPR评估的影响，发现了NPR系统在面对不自然的代码转换时会产生较高的误报率，且在使用自然转换进行评估时性能明显下降。

    

    在本文中，我们研究了保留语义的转换的自然性及其对NPR评估的影响。为了达到这个目的，我们进行了一个两阶段的人类研究，包括(1)与资深软件开发人员的访谈，以建立评估代码转换自然性的第一个具体标准；(2)进行了一项涉及10名开发人员的调查，评估了应用于225个真实世界bug的1178个转换（即原始和转换程序成对的情况）的自然性。我们的研究结果显示，其中接近60%的转换被认为是自然的，20%的转换被认为是不自然的，并且在人类标注者之间有相当高的一致性。此外，不自然的代码转换引入了五个知名NPR系统的稳健性的25.2%误报率。此外，当使用自然转换进行评估时，NPR系统的性能显着下降，即性能下降高达22.9%和23.6%。

    arXiv:2402.11892v1 Announce Type: cross  Abstract: In this paper, we investigate the naturalness of semantic-preserving transformations and their impacts on the evaluation of NPR. To achieve this, we conduct a two-stage human study, including (1) interviews with senior software developers to establish the first concrete criteria for assessing the naturalness of code transformations and (2) a survey involving 10 developers to assess the naturalness of 1178 transformations, i.e., pairs of original and transformed programs, applied to 225 real-world bugs. Our findings reveal that nearly 60% and 20% of these transformations are considered natural and unnatural with substantially high agreement among human annotators. Furthermore, the unnatural code transformations introduce a 25.2% false alarm rate on robustness of five well-known NPR systems. Additionally, the performance of the NPR systems drops notably when evaluated using natural transformations, i.e., a drop of up to 22.9% and 23.6% i
    

