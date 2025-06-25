# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases](https://arxiv.org/abs/2403.16776) | 使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。 |
| [^2] | [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) | 引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。 |
| [^3] | [DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing.](http://arxiv.org/abs/2310.08785) | 本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。 |
| [^4] | [Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages.](http://arxiv.org/abs/2308.16075) | 该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。 |

# 详细

[^1]: Diff-Def: 通过扩散生成的形变场进行有条件的图谱制作

    Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases

    [https://arxiv.org/abs/2403.16776](https://arxiv.org/abs/2403.16776)

    使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。

    

    解剖图谱广泛应用于人口分析。有条件的图谱针对通过特定条件（如人口统计学或病理学）定义的特定子人口，并允许研究与年龄相关的形态学差异等细粒度解剖学差异。现有方法使用基于配准的方法或生成模型，前者无法处理大的解剖学变异，后者可能在训练过程中出现不稳定和幻觉。为了克服这些限制，我们使用潜在扩散模型生成形变场，将一个常规人口图谱转变为代表特定子人口的图谱。通过生成形变场，并将有条件的图谱注册到一组图像附近，我们确保结构的合理性，避免直接图像合成时可能出现的幻觉。我们将我们的方法与几种最先进的方法进行了比较。

    arXiv:2403.16776v1 Announce Type: cross  Abstract: Anatomical atlases are widely used for population analysis. Conditional atlases target a particular sub-population defined via certain conditions (e.g. demographics or pathologies) and allow for the investigation of fine-grained anatomical differences - such as morphological changes correlated with age. Existing approaches use either registration-based methods that are unable to handle large anatomical variations or generative models, which can suffer from training instabilities and hallucinations. To overcome these limitations, we use latent diffusion models to generate deformation fields, which transform a general population atlas into one representing a specific sub-population. By generating a deformation field and registering the conditional atlas to a neighbourhood of images, we ensure structural plausibility and avoid hallucinations, which can occur during direct image synthesis. We compare our method to several state-of-the-art 
    
[^2]: 对齐与提炼：统一和改进领域自适应目标检测

    Align and Distill: Unifying and Improving Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

    引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。

    

    目标检测器通常表现不佳于与其训练集不同的数据。最近，领域自适应目标检测（DAOD）方法已经展示了在应对这一挑战上的强大结果。遗憾的是，我们发现了系统化的基准测试陷阱，这些陷阱对过去的结果提出质疑并阻碍了进一步的进展：（a）由于基线不足导致性能高估，（b）不一致的实现实践阻止了方法的透明比较，（c）由于过时的骨干和基准测试缺乏多样性，导致缺乏普遍性。我们通过引入以下问题来解决这些问题：（1）一个统一的基准测试和实现框架，Align and Distill（ALDI），支持DAOD方法的比较并支持未来发展，（2）一个公平且现代的DAOD训练和评估协议，解决了基准测试的陷阱，（3）一个新的DAOD基准数据集，CFC-DAOD，能够在多样化的真实环境中进行评估。

    arXiv:2403.12029v1 Announce Type: cross  Abstract: Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real
    
[^3]: DeltaSpace:一种用于灵活文本引导图像编辑的语义对齐特征空间

    DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing. (arXiv:2310.08785v1 [cs.CV])

    [http://arxiv.org/abs/2310.08785](http://arxiv.org/abs/2310.08785)

    本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。

    

    文本引导图像编辑面临着训练和推理灵活性的重大挑战。许多文献通过收集大量标注的图像-文本对来从头开始训练文本条件生成模型，这既昂贵又低效。然后，一些利用预训练的视觉语言模型的方法出现了，以避免数据收集，但它们仍然受到基于每个文本提示的优化或推理时的超参数调整的限制。为了解决这些问题，我们调查和确定了一个特定的空间，称为CLIP DeltaSpace，在这个空间中，两个图像的CLIP视觉特征差异与其对应的文本描述的CLIP文本特征差异在语义上是对齐的。基于DeltaSpace，我们提出了一个新颖的框架DeltaEdit，在训练阶段将CLIP视觉特征差异映射到生成模型的潜在空间方向，并从CLIP预测潜在空间方向。

    Text-guided image editing faces significant challenges to training and inference flexibility. Much literature collects large amounts of annotated image-text pairs to train text-conditioned generative models from scratch, which is expensive and not efficient. After that, some approaches that leverage pre-trained vision-language models are put forward to avoid data collection, but they are also limited by either per text-prompt optimization or inference-time hyper-parameters tuning. To address these issues, we investigate and identify a specific space, referred to as CLIP DeltaSpace, where the CLIP visual feature difference of two images is semantically aligned with the CLIP textual feature difference of their corresponding text descriptions. Based on DeltaSpace, we propose a novel framework called DeltaEdit, which maps the CLIP visual feature differences to the latent space directions of a generative model during the training phase, and predicts the latent space directions from the CLIP
    
[^4]: 视觉背景对嘈杂的多模态神经机器翻译的影响：对英印语言的实证研究

    Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages. (arXiv:2308.16075v1 [cs.CL])

    [http://arxiv.org/abs/2308.16075](http://arxiv.org/abs/2308.16075)

    该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。

    

    本研究调查了在神经机器翻译中利用多模态信息的有效性。先前的研究主要关注在资源匮乏的情况下使用多模态数据，而本研究则考察了将图像特征添加到大规模预训练的单模态神经机器翻译系统中的翻译效果。令人惊讶的是，研究发现在这种情况下图像可能是多余的。此外，该研究引入了合成噪声来评估图像是否有助于模型处理文本噪声。在嘈杂的环境中，即使是随机图像，多模态模型在性能上略优于文本模型。实验将英语翻译为印地语、孟加拉语和马拉雅拉姆语，结果显著优于最先进的基准。有趣的是，视觉背景的影响与源文本噪声有所不同：对于非噪声翻译，不使用视觉背景效果最好；对于低噪声，裁剪的图像特征最佳；在高噪声情况下，完整的图像特征效果更好。

    The study investigates the effectiveness of utilizing multimodal information in Neural Machine Translation (NMT). While prior research focused on using multimodal data in low-resource scenarios, this study examines how image features impact translation when added to a large-scale, pre-trained unimodal NMT system. Surprisingly, the study finds that images might be redundant in this context. Additionally, the research introduces synthetic noise to assess whether images help the model deal with textual noise. Multimodal models slightly outperform text-only models in noisy settings, even with random images. The study's experiments translate from English to Hindi, Bengali, and Malayalam, outperforming state-of-the-art benchmarks significantly. Interestingly, the effect of visual context varies with source text noise: no visual context works best for non-noisy translations, cropped image features are optimal for low noise, and full image features work better in high-noise scenarios. This she
    

