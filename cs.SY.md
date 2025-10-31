# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems](https://arxiv.org/abs/2402.03145) | SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。 |

# 详细

[^1]: SafEDMD：一种专为非线性动态系统数据驱动控制而设计的认证学习架构

    SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems

    [https://arxiv.org/abs/2402.03145](https://arxiv.org/abs/2402.03145)

    SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。

    

    Koopman算子作为机器学习动态控制系统的理论基础，其中算子通过扩展动态模态分解（EDMD）启发式近似。在本文中，我们提出了稳定性和认证导向的EDMD（SafEDMD）：一种新颖的基于EDMD的学习架构，它提供了严格的证书，从而以数据驱动的方式生成可靠的替代模型。为了确保SafEDMD的可靠性，我们推导出比例误差界限，这些界限在原点处消失，并且适用于控制任务，从而基于半定规划进行认证控制器设计。我们通过几个基准示例说明了所开发的机制，并强调其相对于现有方法的优势。

    The Koopman operator serves as the theoretical backbone for machine learning of dynamical control systems, where the operator is heuristically approximated by extended dynamic mode decomposition (EDMD). In this paper, we propose Stability- and certificate-oriented EDMD (SafEDMD): a novel EDMD-based learning architecture which comes along with rigorous certificates, resulting in a reliable surrogate model generated in a data-driven fashion. To ensure trustworthiness of SafEDMD, we derive proportional error bounds, which vanish at the origin and are tailored for control tasks, leading to certified controller design based on semi-definite programming. We illustrate the developed machinery by means of several benchmark examples and highlight the advantages over state-of-the-art methods.
    

