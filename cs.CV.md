# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training](https://arxiv.org/abs/2403.03728) | 通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。 |
| [^2] | [Multi-stage Deep Learning Artifact Reduction for Computed Tomography.](http://arxiv.org/abs/2309.00494) | 本论文提出了一种多阶段深度学习伪影减少方法，用于提高计算机断层扫描的图像质量。传统方法通常在重建之后进行处理，而本方法能够根据不同的图像域进行多步骤去伪影，使得相对困难去除的伪影也能够有效消除。 |
| [^3] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |

# 详细

[^1]: 通过自监督预训练在主动学习中弥合多样性与不确定性

    Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training

    [https://arxiv.org/abs/2403.03728](https://arxiv.org/abs/2403.03728)

    通过引入TCM启发式方法，本研究在主动学习中成功结合了多样性采样和不确定性采样策略，解决了冷启动问题并在各种数据水平上表现出色。

    

    本研究探讨了在主动学习中集成基于多样性和基于不确定性的采样策略，特别是在自监督预训练模型的背景下。我们引入了一个称为TCM的简单启发式方法，可以缓解冷启动问题，同时在各种数据水平上保持强大性能。通过首先应用TypiClust进行多样性采样，随后过渡到使用Margin进行不确定性采样，我们的方法有效地结合了两种策略的优势。我们的实验表明，TCM在低数据和高数据情况下始终优于现有方法。

    arXiv:2403.03728v1 Announce Type: cross  Abstract: This study addresses the integration of diversity-based and uncertainty-based sampling strategies in active learning, particularly within the context of self-supervised pre-trained models. We introduce a straightforward heuristic called TCM that mitigates the cold start problem while maintaining strong performance across various data levels. By initially applying TypiClust for diversity sampling and subsequently transitioning to uncertainty sampling with Margin, our approach effectively combines the strengths of both strategies. Our experiments demonstrate that TCM consistently outperforms existing methods across various datasets in both low and high data regimes.
    
[^2]: 计算机断层扫描的多阶段深度学习伪影减少

    Multi-stage Deep Learning Artifact Reduction for Computed Tomography. (arXiv:2309.00494v1 [eess.IV])

    [http://arxiv.org/abs/2309.00494](http://arxiv.org/abs/2309.00494)

    本论文提出了一种多阶段深度学习伪影减少方法，用于提高计算机断层扫描的图像质量。传统方法通常在重建之后进行处理，而本方法能够根据不同的图像域进行多步骤去伪影，使得相对困难去除的伪影也能够有效消除。

    

    在计算机断层扫描中，通过一系列获取的投影图像计算出物体内部结构的图像。这些重建图像的质量对于准确分析至关重要，但是这种质量可能会被各种成像伪影降低。为了提高重建质量，获取的投影图像通常通过由多个去伪影步骤组成的流程进行处理，这些步骤应用于不同的图像域（例如，投影图像的异常值去除和重建图像的去噪）。这些伪影去除方法利用了某些伪影在特定域相对于其他域更容易去除的事实。最近，深度学习方法在计算机断层扫描伪影去除方面取得了有希望的结果。然而，大多数现有的计算机断层扫描深度学习方法都是在重建之后作为后处理方法应用的。因此，在重建域相对困难去除的伪影可能无法有效去除。

    In Computed Tomography (CT), an image of the interior structure of an object is computed from a set of acquired projection images. The quality of these reconstructed images is essential for accurate analysis, but this quality can be degraded by a variety of imaging artifacts. To improve reconstruction quality, the acquired projection images are often processed by a pipeline consisting of multiple artifact-removal steps applied in various image domains (e.g., outlier removal on projection images and denoising of reconstruction images). These artifact-removal methods exploit the fact that certain artifacts are easier to remove in a certain domain compared with other domains.  Recently, deep learning methods have shown promising results for artifact removal for CT images. However, most existing deep learning methods for CT are applied as a post-processing method after reconstruction. Therefore, artifacts that are relatively difficult to remove in the reconstruction domain may not be effec
    
[^3]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    

