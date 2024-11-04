# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models](https://arxiv.org/abs/2404.01318) | JailbreakBench是一个用于对抗大型语言模型越狱的开放基准，提供新的数据集、对抗提示和评估框架。 |
| [^2] | [Group Privacy Amplification and Unified Amplification by Subsampling for R\'enyi Differential Privacy](https://arxiv.org/abs/2403.04867) | 该论文提出了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。 |

# 详细

[^1]: JailbreakBench: 一个用于对抗大型语言模型越狱的开放鲁棒性基准

    JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models

    [https://arxiv.org/abs/2404.01318](https://arxiv.org/abs/2404.01318)

    JailbreakBench是一个用于对抗大型语言模型越狱的开放基准，提供新的数据集、对抗提示和评估框架。

    

    越狱攻击会导致大型语言模型生成有害、不道德或令人反感的内容。评估这些攻击存在许多挑战，当前的基准和评估技术并未充分解决。为了解决这些挑战，我们引入了JailbreakBench，一个开源基准，包括具有100个独特行为的新越狱数据集（称为JBB-Behaviors）、一组最先进的对抗提示（称为越狱工件）和一个标准化评估框架。

    arXiv:2404.01318v1 Announce Type: cross  Abstract: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) a new jailbreaking dataset containing 100 unique behaviors, which we call JBB-Behaviors; (2) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (3) a standardized evaluation framework that i
    
[^2]: 组隐私放大和子抽样的Rényi差分隐私统一放大

    Group Privacy Amplification and Unified Amplification by Subsampling for R\'enyi Differential Privacy

    [https://arxiv.org/abs/2403.04867](https://arxiv.org/abs/2403.04867)

    该论文提出了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。

    

    差分隐私(DP)具有多种理想属性，如对后处理的鲁棒性、组隐私和通过子抽样放大，这些属性可以相互独立推导。我们的目标是确定是否通过联合考虑这些属性中的多个可以获得更强的隐私保证。为此，我们专注于组隐私和通过子抽样放大的组合。为了提供适合机器学习算法的保证，我们在Rényi-DP框架中进行了分析，这比$(\epsilon,\delta)$-DP具有更有利的组合属性。作为这个分析的一部分，我们开发了一个统一的框架，用于为Rényi-DP推导通过子抽样的放大保证，这是首个针对隐私核算方法的框架，也具有独立的重要性。我们发现，它不仅让我们改进和泛化现有的放大结果。

    arXiv:2403.04867v1 Announce Type: cross  Abstract: Differential privacy (DP) has various desirable properties, such as robustness to post-processing, group privacy, and amplification by subsampling, which can be derived independently of each other. Our goal is to determine whether stronger privacy guarantees can be obtained by considering multiple of these properties jointly. To this end, we focus on the combination of group privacy and amplification by subsampling. To provide guarantees that are amenable to machine learning algorithms, we conduct our analysis in the framework of R\'enyi-DP, which has more favorable composition properties than $(\epsilon,\delta)$-DP. As part of this analysis, we develop a unified framework for deriving amplification by subsampling guarantees for R\'enyi-DP, which represents the first such framework for a privacy accounting method and is of independent interest. We find that it not only lets us improve upon and generalize existing amplification results 
    

