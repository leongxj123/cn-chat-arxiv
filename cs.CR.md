# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Privacy with Non-Private Pre-Processing](https://arxiv.org/abs/2403.13041) | 提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限 |
| [^2] | [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2402.17840) | 研究揭示了检索增强生成系统中的数据泄露风险，指出对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据，并设计了攻击对生产RAG模型GPTs造成数据存储泄漏。 |
| [^3] | [The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/abs/2309.01243) | 本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。 |
| [^4] | [FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs.](http://arxiv.org/abs/2306.04959) | 本文介绍了一个名为FedMLSecurity的基准测试，它可以模拟在联邦学习中可能出现的对抗攻击并提供相应的防御策略。该测试对各种机器学习模型和联合优化器都可以适用，并且能够轻松应用于大规模语言模型中。 |

# 详细

[^1]: 具有非私密预处理的可证明隐私

    Provable Privacy with Non-Private Pre-Processing

    [https://arxiv.org/abs/2403.13041](https://arxiv.org/abs/2403.13041)

    提出了一个框架，能够评估非私密数据相关预处理算法引起的额外隐私成本，并利用平滑DP和预处理算法的有界敏感性建立整体隐私保证的上限

    

    当分析差分私密（DP）机器学习管道时，通常会忽略数据相关的预处理的潜在隐私成本。在这项工作中，我们提出了一个通用框架，用于评估由非私密数据相关预处理算法引起的额外隐私成本。我们的框架通过利用两个新的技术概念建立了整体隐私保证的上限：一种称为平滑DP的DP变体以及预处理算法的有界敏感性。

    arXiv:2403.13041v1 Announce Type: cross  Abstract: When analysing Differentially Private (DP) machine learning pipelines, the potential privacy cost of data-dependent pre-processing is frequently overlooked in privacy accounting. In this work, we propose a general framework to evaluate the additional privacy cost incurred by non-private data-dependent pre-processing algorithms. Our framework establishes upper bounds on the overall privacy guarantees by utilising two new technical notions: a variant of DP termed Smooth DP and the bounded sensitivity of the pre-processing algorithms. In addition to the generic framework, we provide explicit overall privacy guarantees for multiple data-dependent pre-processing algorithms, such as data imputation, quantization, deduplication and PCA, when used in combination with several DP algorithms. Notably, this framework is also simple to implement, allowing direct integration into existing DP pipelines.
    
[^2]: 遵循我的指示并说出真相：来自检索增强生成系统的可扩展数据提取

    Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems

    [https://arxiv.org/abs/2402.17840](https://arxiv.org/abs/2402.17840)

    研究揭示了检索增强生成系统中的数据泄露风险，指出对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据，并设计了攻击对生产RAG模型GPTs造成数据存储泄漏。

    

    检索增强生成（RAG）通过在测试时将外部知识纳入预训练模型，从而实现定制适应，提升了模型性能。本研究探讨了Retrieval-In-Context RAG语言模型（LMs）中的数据泄露风险。我们展示了当对使用指令调整的LMs构建的RAG系统进行提示注入时，对手可以利用LMs的指示遵循能力轻松地从数据存储中直接提取文本数据。这种漏洞存在于覆盖Llama2、Mistral/Mixtral、Vicuna、SOLAR、WizardLM、Qwen1.5和Platypus2等多种现代LMs的广泛范围内，并且随着模型规模的扩大，利用能力加剧。将研究扩展到生产RAG模型GPTs，我们设计了一种攻击，可以在对25个随机选择的定制GPTs施加最多2个查询时以100%成功率导致数据存储泄漏，并且我们能够以77,000字的书籍中的文本数据的提取率为41%，以及在含有1,569,00词的语料库中的文本数据的提取率为3%。

    arXiv:2402.17840v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,00
    
[^3]: 正态分布不可区分性谱及其在隐私保护机器学习中的应用

    The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning

    [https://arxiv.org/abs/2309.01243](https://arxiv.org/abs/2309.01243)

    本文提出了正态分布不可区分性谱定理 (NDIS Theorem)，旨在利用查询本身的随机性改进随机化机器学习查询的差分隐私机制。

    

    要实现差分隐私(DP)，通常需要随机化基础查询的输出。在大数据分析中，人们经常使用随机化草图/聚合算法来使处理高维数据变得可行。直观地，这样的机器学习(ML)算法应该提供一些固有的隐私性，但现有的大部分DP机制并没有利用这种固有的随机性，导致潜在的多余噪音。我们工作的动机问题是：(如何)可以通过利用查询本身的随机性来提高随机化ML查询的DP机制的效用？为了给出积极的答案，我们证明了正态分布不可区分性谱定理(简称为NDIS定理)，这是一个具有深远实际影响的理论结果。总的来说，NDIS是一个用于$(\epsilon,\delta)$-不可区分性谱(简称为$

    arXiv:2309.01243v2 Announce Type: replace-cross  Abstract: To achieve differential privacy (DP) one typically randomizes the output of the underlying query. In big data analytics, one often uses randomized sketching/aggregation algorithms to make processing high-dimensional data tractable. Intuitively, such machine learning (ML) algorithms should provide some inherent privacy, yet most if not all existing DP mechanisms do not leverage this inherent randomness, resulting in potentially redundant noising.   The motivating question of our work is:   (How) can we improve the utility of DP mechanisms for randomized ML queries, by leveraging the randomness of the query itself?   Towards a (positive) answer, we prove the Normal Distributions Indistinguishability Spectrum Theorem (in short, NDIS Theorem), a theoretical result with far-reaching practical implications. In a nutshell, NDIS is a closed-form analytic computation for the $(\epsilon,\delta)$-indistinguishability-spectrum (in short, $
    
[^4]: FedMLSecurity：联邦学习与LLMs中攻击与防御的基准测试

    FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs. (arXiv:2306.04959v1 [cs.CR])

    [http://arxiv.org/abs/2306.04959](http://arxiv.org/abs/2306.04959)

    本文介绍了一个名为FedMLSecurity的基准测试，它可以模拟在联邦学习中可能出现的对抗攻击并提供相应的防御策略。该测试对各种机器学习模型和联合优化器都可以适用，并且能够轻松应用于大规模语言模型中。

    

    本文介绍了FedMLSecurity，这是一个在联邦学习（FL）中模拟对抗攻击和相应防御机制的基准测试。作为开源库FedML的一个重要模块，FedMLSecurity增强了FedML的安全评估能力。FedMLSecurity包含两个主要组件：FedMLAttacker模拟在FL训练中注入的攻击，而FedMLDefender则模拟旨在减轻攻击影响的防御策略。FedMLSecurity是开源的，可适用于各种机器学习模型（例如逻辑回归，ResNet，GAN等）和联合优化器（例如FedAVG，FedOPT，FedNOVA等）。本文的实验评估还展示了将FedMLSecurity轻松应用于LLMs的便利性，进一步强化了其各种场景下的通用性和实用性。

    This paper introduces FedMLSecurity, a benchmark that simulates adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances the security assessment capacity of FedML. FedMLSecurity comprises two principal components: FedMLAttacker, which simulates attacks injected into FL training, and FedMLDefender, which emulates defensive strategies designed to mitigate the impacts of the attacks. FedMLSecurity is open-sourced 1 and is customizable to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). Experimental evaluations in this paper also demonstrate the ease of application of FedMLSecurity to Large Language Models (LLMs), further reinforcing its versatility and practical utility in various scenarios.
    

