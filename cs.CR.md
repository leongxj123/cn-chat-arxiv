# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^2] | [Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach](https://arxiv.org/abs/2403.10112) | 本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。 |
| [^3] | [Don't Forget What I did?: Assessing Client Contributions in Federated Learning](https://arxiv.org/abs/2403.07151) | 提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。 |
| [^4] | [How Does Selection Leak Privacy: Revisiting Private Selection and Improved Results for Hyper-parameter Tuning](https://arxiv.org/abs/2402.13087) | 本论文探讨了超参数调整中的隐私性问题，发现当前的隐私分析在一般情况下是紧密的，但在特定的超参数调整问题上则不再成立，并通过隐私审计揭示了当前理论隐私界与实证之间的显著差距。 |
| [^5] | [Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings.](http://arxiv.org/abs/2308.11804) | 该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。 |

# 详细

[^1]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^2]: 单智能体与多智能体的私密主动感知：深度神经进化方法

    Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach

    [https://arxiv.org/abs/2403.10112](https://arxiv.org/abs/2403.10112)

    本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。

    

    本文关注存在窥视者情况下的主动假设测试中的一个集中式问题和一个分散式问题。针对包括单个合法智能体的集中式问题，我们提出了基于神经进化（NE）的新框架；而针对分散式问题，我们开发了一种新颖的基于NE的方法，用于解决协作多智能体任务，这种方法有趣地保持了单一智能体NE的所有计算优势。通过对无线传感器网络上异常检测示例用例中的数值实验，验证了所提出的EAHT方法优于传统的主动假设测试策略以及基于学习的方法。

    arXiv:2403.10112v1 Announce Type: new  Abstract: In this paper, we focus on one centralized and one decentralized problem of active hypothesis testing in the presence of an eavesdropper. For the centralized problem including a single legitimate agent, we present a new framework based on NeuroEvolution (NE), whereas, for the decentralized problem, we develop a novel NE-based method for solving collaborative multi-agent tasks, which interestingly maintains all computational benefits of single-agent NE. The superiority of the proposed EAHT approaches over conventional active hypothesis testing policies, as well as learning-based methods, is validated through numerical investigations in an example use case of anomaly detection over wireless sensor networks.
    
[^3]: 不要忘记我做的事：评估联邦学习中的客户贡献

    Don't Forget What I did?: Assessing Client Contributions in Federated Learning

    [https://arxiv.org/abs/2403.07151](https://arxiv.org/abs/2403.07151)

    提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。

    

    联邦学习（FL）是一种协作机器学习（ML）方法，多个客户参与训练ML模型，而不暴露私人数据。公平准确评估客户贡献在FL中是一个重要问题，以促进激励分配并鼓励多样化客户参与统一模型训练。本文提出了一个历史感知的博弈理论框架FLContrib，用于评估在每个FL训练时期中的（潜在非独立同分布）客户参与。

    arXiv:2403.07151v1 Announce Type: cross  Abstract: Federated Learning (FL) is a collaborative machine learning (ML) approach, where multiple clients participate in training an ML model without exposing the private data. Fair and accurate assessment of client contributions is an important problem in FL to facilitate incentive allocation and encouraging diverse clients to participate in a unified model training. Existing methods for assessing client contribution adopts co-operative game-theoretic concepts, such as Shapley values, but under simplified assumptions. In this paper, we propose a history-aware game-theoretic framework, called FLContrib, to assess client contributions when a subset of (potentially non-i.i.d.) clients participate in each epoch of FL training. By exploiting the FL training process and linearity of Shapley value, we develop FLContrib that yields a historical timeline of client contributions as FL training progresses over epochs. Additionally, to assess client cont
    
[^4]: 选择如何泄漏隐私：重新审视私有选择及超参数调整的改进结果

    How Does Selection Leak Privacy: Revisiting Private Selection and Improved Results for Hyper-parameter Tuning

    [https://arxiv.org/abs/2402.13087](https://arxiv.org/abs/2402.13087)

    本论文探讨了超参数调整中的隐私性问题，发现当前的隐私分析在一般情况下是紧密的，但在特定的超参数调整问题上则不再成立，并通过隐私审计揭示了当前理论隐私界与实证之间的显著差距。

    

    我们研究了在超参数调整中保证差分隐私(DP)的问题，这是机器学习中一个关键的过程，涉及从几个运行中选择最佳的过程。与许多私有算法（包括普遍存在的DP-SGD）不同，调整的隐私影响仍然不够了解。最近的研究提出了一个通用的私有解决方案用于调整过程，然而一个根本的问题仍然存在：当前解决方案的隐私界是否紧密？本文对这个问题提出了积极和消极的答案。最初，我们提供的研究证实了当前的隐私分析在一般意义上确实是紧密的。然而，当我们专门研究超参数调整问题时，这种紧密性则不再成立。首先，通过对调整过程进行隐私审计来证明了这一点。我们的研究结果突显了当前理论隐私界与实证之间存在重大差距。

    arXiv:2402.13087v1 Announce Type: new  Abstract: We study the problem of guaranteeing Differential Privacy (DP) in hyper-parameter tuning, a crucial process in machine learning involving the selection of the best run from several. Unlike many private algorithms, including the prevalent DP-SGD, the privacy implications of tuning remain insufficiently understood. Recent works propose a generic private solution for the tuning process, yet a fundamental question still persists: is the current privacy bound for this solution tight?   This paper contributes both positive and negative answers to this question. Initially, we provide studies affirming the current privacy analysis is indeed tight in a general sense. However, when we specifically study the hyper-parameter tuning problem, such tightness no longer holds. This is first demonstrated by applying privacy audit on the tuning process. Our findings underscore a substantial gap between the current theoretical privacy bound and the empirica
    
[^5]: 这不是一个苹果：多模态嵌入中的对抗幻觉

    Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings. (arXiv:2308.11804v1 [cs.CR])

    [http://arxiv.org/abs/2308.11804](http://arxiv.org/abs/2308.11804)

    该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。

    

    多模态编码器将图像、声音、文本、视频等映射到一个单一的嵌入空间中，通过对齐不同模态的表示（例如将一张狗的图像与一种叫声相关联）。我们展示了多模态嵌入可以受到一种我们称之为“对抗幻觉”的攻击。给定任意模态的输入，对手可以扰动它，使其嵌入接近于另一模态中任意对手选择的输入的嵌入。幻觉使对手能够将任意图像与任意文本、任意文本与任意声音等进行对齐。对抗幻觉利用了嵌入空间中的接近性，因此与下游任务无关。使用ImageBind嵌入，我们演示了在没有具体下游任务知识的情况下，通过对抗性对齐的输入如何误导图像生成、文本生成和零样例分类。

    Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.  Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.
    

