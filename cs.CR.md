# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Smooth Sensitivity for Learning Differentially-Private yet Accurate Rule Lists](https://arxiv.org/abs/2403.13848) | 通过建立Gini不纯度的平滑敏感度并将其应用于提出DP贪婪规则列表算法，本文改善了差异保护模型的准确性问题。 |
| [^2] | [DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers](https://arxiv.org/abs/2402.16914) | 将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测 |

# 详细

[^1]: 用于学习差异保护但准确规则列表的平滑敏感度

    Smooth Sensitivity for Learning Differentially-Private yet Accurate Rule Lists

    [https://arxiv.org/abs/2403.13848](https://arxiv.org/abs/2403.13848)

    通过建立Gini不纯度的平滑敏感度并将其应用于提出DP贪婪规则列表算法，本文改善了差异保护模型的准确性问题。

    

    差异保护（DP）机制可以嵌入到机器学习算法的设计中，以保护所得模型免受隐私泄露的影响，尽管这通常伴随着明显的准确性损失。本文旨在通过建立Gini不纯度的平滑敏感度并利用这一特性来提出一个DP贪婪规则列表算法，以改善这种权衡。我们的理论分析和实验结果表明，集成平滑敏感度的DP规则列表模型具有比使用全局敏感度的其他DP框架更高的准确性。

    arXiv:2403.13848v1 Announce Type: cross  Abstract: Differentially-private (DP) mechanisms can be embedded into the design of a machine learningalgorithm to protect the resulting model against privacy leakage, although this often comes with asignificant loss of accuracy. In this paper, we aim at improving this trade-off for rule lists modelsby establishing the smooth sensitivity of the Gini impurity and leveraging it to propose a DP greedyrule list algorithm. In particular, our theoretical analysis and experimental results demonstrate thatthe DP rule lists models integrating smooth sensitivity have higher accuracy that those using otherDP frameworks based on global sensitivity.
    
[^2]: DrAttack: 提示分解和重构使强大的LLM越狱者

    DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers

    [https://arxiv.org/abs/2402.16914](https://arxiv.org/abs/2402.16914)

    将恶意提示分解为独立的子提示使得LLM越狱攻击更难被检测

    

    本文发现将恶意提示分解为独立的子提示能够有效模糊其潜在的恶意意图，使之以片段化、不易检测的形式呈现，从而解决了这些局限性。我们引入了一个用于越狱攻击的自动提示分解和重构框架（DrAttack）。DrAttack包括三个关键组件：(a) 将原始提示进行“分解”为子提示，(b) 通过上下文学习中的语义上相似但隐含的“重构”这些子提示

    arXiv:2402.16914v1 Announce Type: cross  Abstract: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but h
    

