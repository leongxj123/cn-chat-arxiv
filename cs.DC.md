# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness, Efficiency, or Privacy: Pick Two in Machine Learning](https://arxiv.org/abs/2312.14712) | 该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。 |

# 详细

[^1]: 机器学习中的鲁棒性、效率或隐私：只能选两样

    Robustness, Efficiency, or Privacy: Pick Two in Machine Learning

    [https://arxiv.org/abs/2312.14712](https://arxiv.org/abs/2312.14712)

    该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。

    

    机器学习（ML）应用的成功依赖于庞大的数据集和分布式架构，随着它们的增长，这些架构带来了重大挑战。在真实世界的场景中，数据通常包含敏感信息，数据污染和硬件故障等问题很常见。确保隐私和鲁棒性对于ML在公共生活中的广泛应用至关重要。本文从理论和实证角度研究了在分布式ML架构中实现这些目标所带来的成本。我们概述了分布式ML中隐私和鲁棒性的含义，并阐明了如何单独高效实现它们。然而，我们认为整合这两个目标会在计算效率上有显著的折衷。简而言之，传统的噪声注入通过隐藏毒害输入来损害准确性，而加密方法与防毒防御相冲突，因为它们是非线性的。

    arXiv:2312.14712v2 Announce Type: replace  Abstract: The success of machine learning (ML) applications relies on vast datasets and distributed architectures which, as they grow, present major challenges. In real-world scenarios, where data often contains sensitive information, issues like data poisoning and hardware failures are common. Ensuring privacy and robustness is vital for the broad adoption of ML in public life. This paper examines the costs associated with achieving these objectives in distributed ML architectures, from both theoretical and empirical perspectives. We overview the meanings of privacy and robustness in distributed ML, and clarify how they can be achieved efficiently in isolation. However, we contend that the integration of these two objectives entails a notable compromise in computational efficiency. In short, traditional noise injection hurts accuracy by concealing poisoned inputs, while cryptographic methods clash with poisoning defenses due to their non-line
    

