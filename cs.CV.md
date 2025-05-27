# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases](https://arxiv.org/abs/2403.16776) | 使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。 |
| [^2] | [REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning](https://arxiv.org/abs/2403.13522) | 本文提出了REAL方法，通过构建双流基础预训练和表示增强蒸馏过程来增强提取器的表示，从而解决了无范例类增量学习中的遗忘问题。 |
| [^3] | [GUARD: Role-playing to Generate Natural-language Jailbreakings to Test Guideline Adherence of Large Language Models](https://arxiv.org/abs/2402.03299) | 本论文提出了一个通过角色扮演的系统，可以生成自然语言越狱，用于测试大型语言模型的指南遵循情况。系统通过收集现有越狱并将其组织成知识图来生成新的越狱，证明了其高效性和有效性。 |
| [^4] | [FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction](https://arxiv.org/abs/2312.03187) | 开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。 |
| [^5] | [RefinedFields: Radiance Fields Refinement for Unconstrained Scenes](https://arxiv.org/abs/2312.00639) | RefinedFields是第一种利用预训练模型改善无约束场景建模的方法。通过优化指导和交替训练过程，该方法能够从真实世界图像的先验条件中提取更丰富的细节，并在新视角合成任务中优于以往的方法。 |

# 详细

[^1]: Diff-Def: 通过扩散生成的形变场进行有条件的图谱制作

    Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases

    [https://arxiv.org/abs/2403.16776](https://arxiv.org/abs/2403.16776)

    使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。

    

    解剖图谱广泛应用于人口分析。有条件的图谱针对通过特定条件（如人口统计学或病理学）定义的特定子人口，并允许研究与年龄相关的形态学差异等细粒度解剖学差异。现有方法使用基于配准的方法或生成模型，前者无法处理大的解剖学变异，后者可能在训练过程中出现不稳定和幻觉。为了克服这些限制，我们使用潜在扩散模型生成形变场，将一个常规人口图谱转变为代表特定子人口的图谱。通过生成形变场，并将有条件的图谱注册到一组图像附近，我们确保结构的合理性，避免直接图像合成时可能出现的幻觉。我们将我们的方法与几种最先进的方法进行了比较。

    arXiv:2403.16776v1 Announce Type: cross  Abstract: Anatomical atlases are widely used for population analysis. Conditional atlases target a particular sub-population defined via certain conditions (e.g. demographics or pathologies) and allow for the investigation of fine-grained anatomical differences - such as morphological changes correlated with age. Existing approaches use either registration-based methods that are unable to handle large anatomical variations or generative models, which can suffer from training instabilities and hallucinations. To overcome these limitations, we use latent diffusion models to generate deformation fields, which transform a general population atlas into one representing a specific sub-population. By generating a deformation field and registering the conditional atlas to a neighbourhood of images, we ensure structural plausibility and avoid hallucinations, which can occur during direct image synthesis. We compare our method to several state-of-the-art 
    
[^2]: REAL：用于无范例类增量学习的表示增强分析学习

    REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning

    [https://arxiv.org/abs/2403.13522](https://arxiv.org/abs/2403.13522)

    本文提出了REAL方法，通过构建双流基础预训练和表示增强蒸馏过程来增强提取器的表示，从而解决了无范例类增量学习中的遗忘问题。

    

    无范例的类增量学习(EFCIL)旨在减轻类增量学习中的灾难性遗忘，而没有可用的历史数据。与存储历史样本的回放式CIL相比，EFCIL在无范例约束下更容易遗忘。在本文中，受最近发展的基于分析学习(AL)的CIL的启发，我们提出了一种用于EFCIL的表示增强分析学习(REAL)。REAL构建了一个双流基础预训练(DS-BPT)和一个表示增强蒸馏(RED)过程，以增强提取器的表示。DS-BPT在监督学习和自监督对比学习(SSCL)两个流中预训练模型，用于基础知识提取。RED过程将监督知识提炼到SSCL预训练骨干部分，促进后续的基于AL的CIL，将CIL转换为递归最小化学习

    arXiv:2403.13522v1 Announce Type: new  Abstract: Exemplar-free class-incremental learning (EFCIL) aims to mitigate catastrophic forgetting in class-incremental learning without available historical data. Compared with its counterpart (replay-based CIL) that stores historical samples, the EFCIL suffers more from forgetting issues under the exemplar-free constraint. In this paper, inspired by the recently developed analytic learning (AL) based CIL, we propose a representation enhanced analytic learning (REAL) for EFCIL. The REAL constructs a dual-stream base pretraining (DS-BPT) and a representation enhancing distillation (RED) process to enhance the representation of the extractor. The DS-BPT pretrains model in streams of both supervised learning and self-supervised contrastive learning (SSCL) for base knowledge extraction. The RED process distills the supervised knowledge to the SSCL pretrained backbone and facilitates a subsequent AL-basd CIL that converts the CIL to a recursive least
    
[^3]: GUARD: 通过角色扮演生成自然语言越狱来测试大型语言模型遵循指南的合规性

    GUARD: Role-playing to Generate Natural-language Jailbreakings to Test Guideline Adherence of Large Language Models

    [https://arxiv.org/abs/2402.03299](https://arxiv.org/abs/2402.03299)

    本论文提出了一个通过角色扮演的系统，可以生成自然语言越狱，用于测试大型语言模型的指南遵循情况。系统通过收集现有越狱并将其组织成知识图来生成新的越狱，证明了其高效性和有效性。

    

    发现绕过大型语言模型（LLM）的安全过滤和有害回应的"越狱"已经鼓励社区采取安全措施。其中一个主要的安全措施是在发布之前用越狱主动测试LLM。因此，这样的测试将需要一种能够大规模且高效地生成越狱的方法。本文在追随一种新颖而直观的策略下，以人类生成的方式来生成越狱。我们提出了一个角色扮演系统，将四种不同角色分配给用户LLM，以便协作生成新的越狱。此外，我们收集现有的越狱，并通过句子逐句进行聚类频率和语义模式的划分，将它们分成不同的独立特征。我们将这些特征组织成一个知识图，使其更易于访问和检索。我们的角色系统将利用这个知识图来生成新的越狱，证明了其有效性。

    The discovery of "jailbreaks" to bypass safety filters of Large Language Models (LLMs) and harmful responses have encouraged the community to implement safety measures. One major safety measure is to proactively test the LLMs with jailbreaks prior to the release. Therefore, such testing will require a method that can generate jailbreaks massively and efficiently. In this paper, we follow a novel yet intuitive strategy to generate jailbreaks in the style of the human generation. We propose a role-playing system that assigns four different roles to the user LLMs to collaborate on new jailbreaks. Furthermore, we collect existing jailbreaks and split them into different independent characteristics using clustering frequency and semantic patterns sentence by sentence. We organize these characteristics into a knowledge graph, making them more accessible and easier to retrieve. Our system of different roles will leverage this knowledge graph to generate new jailbreaks, which have proved effec
    
[^4]: FERGI：来自自发面部表情反应的文本到图像生成用户偏好的自动注释

    FERGI: Automatic Annotation of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction

    [https://arxiv.org/abs/2312.03187](https://arxiv.org/abs/2312.03187)

    开发了一种从用户自发面部表情反应中自动注释用户对生成图像偏好的方法，发现多个面部动作单元与用户对生成图像的评估高度相关，可用于通过这些面部动作单元区分图像对并自动标注用户偏好。

    

    研究人员提出使用人类偏好反馈数据来微调文本到图像生成模型。然而，由于其依赖于手动注释，人类反馈收集的可扩展性受到限制。因此，我们开发并测试了一种方法，从用户的自发面部表情反应中自动注释其对生成图像的偏好。我们收集了一个面部表情反应到生成图像（FERGI）的数据集，并展示了多个面部运动单元（AUs）的激活与用户对生成图像的评估高度相关。具体来说，AU4（眉毛下垂者）反映了对生成图像的负面评价，而AU12（嘴角拉动者）反映了正面评价。这两者在两个方面都很有用。首先，我们可以准确地使用这些AU响应存在实质差异的图像对之间自动注释用户偏好。

    arXiv:2312.03187v2 Announce Type: replace-cross  Abstract: Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically annotate user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. Specifically, AU4 (brow lowerer) is reflective of negative evaluations of the generated image whereas AU12 (lip corner puller) is reflective of positive evaluations. These can be useful in two ways. Firstly, we can automatically annotate user preferences between image pairs with substantial difference in these AU responses with an accuracy sig
    
[^5]: RefinedFields: 对无约束场景的辐射场细化

    RefinedFields: Radiance Fields Refinement for Unconstrained Scenes

    [https://arxiv.org/abs/2312.00639](https://arxiv.org/abs/2312.00639)

    RefinedFields是第一种利用预训练模型改善无约束场景建模的方法。通过优化指导和交替训练过程，该方法能够从真实世界图像的先验条件中提取更丰富的细节，并在新视角合成任务中优于以往的方法。

    

    从无约束的图像中建模大场景被证明是计算机视觉中的一个重大挑战。现有方法处理野外场景建模是在封闭的环境中，没有对从真实世界图像获得的先验条件进行约束。我们提出了RefinedFields，这是我们所知的第一种利用预训练模型来改善野外场景建模的方法。我们使用预训练网络通过优化指导使用交替训练过程来细化K-Planes表示。我们进行了大量实验证实我们方法在合成数据和真实旅游照片集上的优点。RefinedFields增强了渲染场景的细节，优于以往在野外进行新视角合成任务的工作。我们的项目页面可以在https://refinedfields.github.io找到。

    Modeling large scenes from unconstrained images has proven to be a major challenge in computer vision. Existing methods tackling in-the-wild scene modeling operate in closed-world settings, where no conditioning on priors acquired from real-world images is present. We propose RefinedFields, which is, to the best of our knowledge, the first method leveraging pre-trained models to improve in-the-wild scene modeling. We employ pre-trained networks to refine K-Planes representations via optimization guidance using an alternating training procedure. We carry out extensive experiments and verify the merit of our method on synthetic data and real tourism photo collections. RefinedFields enhances rendered scenes with richer details and outperforms previous work on the task of novel view synthesis in the wild. Our project page can be found at https://refinedfields.github.io .
    

