# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering](https://arxiv.org/abs/2403.05053) | 本文提出了PrimeComposer，一种更快的逐步组合扩散方式，用于图像合成，主要专注于前景生成，从而解决了合成中的凝聚混乱和外观信息丢失问题，并避免了不必要的背景生成导致的前景生成质量下降。 |
| [^2] | [Segment, Select, Correct: A Framework for Weakly-Supervised Referring Segmentation.](http://arxiv.org/abs/2310.13479) | 这项研究提出了一个弱监督框架，通过将指代图像分割任务分解为获取实例掩模、选择正确掩模和纠正错误掩模的三个步骤，填补了弱监督和零样本方法在性能上的差距。 |
| [^3] | [Atlas-Based Interpretable Age Prediction.](http://arxiv.org/abs/2307.07439) | 本研究提出了一种基于图谱的可解释年龄预测方法，利用全身图像研究了各个身体部位的年龄相关变化。通过使用解释性方法和配准技术，确定了最能预测年龄的身体区域，并创下了整个身体年龄预测的最新水平。研究结果表明，脊柱、本原性背部肌肉和心脏区域是最重要的关注领域。 |
| [^4] | [MMBench: Is Your Multi-modal Model an All-around Player?.](http://arxiv.org/abs/2307.06281) | MMBench是一个新型的多模态基准测试，旨在解决大型视觉语言模型评估的挑战，通过开发全面的评估流程和精心策划的数据集进行细粒度能力评估。 |
| [^5] | [Efficient Quantization-aware Training with Adaptive Coreset Selection.](http://arxiv.org/abs/2306.07215) | 本研究提出了一种用于改善量化感知训练的训练效率的方法，通过核心集选择和两个重要性指标来选择训练数据的子集。 |
| [^6] | [DiracDiffusion: Denoising and Incremental Reconstruction with Assured Data-Consistency.](http://arxiv.org/abs/2303.14353) | DiracDiffusion是一种新的逆问题求解框架，可以应用于图像去噪和增量重建，并保证数据一致性。 |

# 详细

[^1]: PrimeComposer：用于图像合成的快速逐步组合扩散方法和带有注意力引导的技术

    PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering

    [https://arxiv.org/abs/2403.05053](https://arxiv.org/abs/2403.05053)

    本文提出了PrimeComposer，一种更快的逐步组合扩散方式，用于图像合成，主要专注于前景生成，从而解决了合成中的凝聚混乱和外观信息丢失问题，并避免了不必要的背景生成导致的前景生成质量下降。

    

    图像合成涉及将给定对象无缝地整合到特定的视觉环境中。目前无需训练的方法依赖于从几个采样器中组合注意力权重来引导生成器。然而，由于这些权重来自不同的上下文，它们的组合导致在合成中凝聚混乱和外观信息的丢失。在该任务中，它们过多关注背景生成，即使在这项任务中是不必要的，这些问题恶化。这不仅减慢了推理速度，还损害了前景生成质量。此外，这些方法还在过渡区域引入了不需要的伪影。在本文中，我们将图像合成形式化为一项基于主题的局部编辑任务，仅专注于前景生成。在每一步中，编辑后的前景与噪声背景相结合，以保持场景一致性。为了解决剩下的问题，我们提出了PrimeComposer，一种更快的tr

    arXiv:2403.05053v1 Announce Type: cross  Abstract: Image composition involves seamlessly integrating given objects into a specific visual context. The current training-free methods rely on composing attention weights from several samplers to guide the generator. However, since these weights are derived from disparate contexts, their combination leads to coherence confusion in synthesis and loss of appearance information. These issues worsen with their excessive focus on background generation, even when unnecessary in this task. This not only slows down inference but also compromises foreground generation quality. Moreover, these methods introduce unwanted artifacts in the transition area. In this paper, we formulate image composition as a subject-based local editing task, solely focusing on foreground generation. At each step, the edited foreground is combined with the noisy background to maintain scene consistency. To address the remaining issues, we propose PrimeComposer, a faster tr
    
[^2]: 分段、选择、纠正：一种弱监督指代分割的框架

    Segment, Select, Correct: A Framework for Weakly-Supervised Referring Segmentation. (arXiv:2310.13479v1 [cs.CV])

    [http://arxiv.org/abs/2310.13479](http://arxiv.org/abs/2310.13479)

    这项研究提出了一个弱监督框架，通过将指代图像分割任务分解为获取实例掩模、选择正确掩模和纠正错误掩模的三个步骤，填补了弱监督和零样本方法在性能上的差距。

    

    指代图像分割（RIS）是通过自然语言句子在图像中识别对象的一个具有挑战性的任务，目前主要通过监督学习来解决。然而，收集指代标注掩模是一个耗时的过程，现有的弱监督和零样本方法在性能上远远不及完全监督学习的方法。为了填补性能差距，我们提出了一种新的弱监督框架，通过将RIS分解成三个步骤进行处理：获取被提及指令中的对象的实例掩模（分段），使用零样本学习来选择给定指令的潜在正确掩模（选择），并通过引导模型来修正零样本选择的错误（纠正）。在我们的实验中，仅使用前两个步骤（零样本分段和选择）比其他零样本基线提高了多达19%的性能。

    Referring Image Segmentation (RIS) - the problem of identifying objects in images through natural language sentences - is a challenging task currently mostly solved through supervised learning. However, while collecting referred annotation masks is a time-consuming process, the few existing weakly-supervised and zero-shot approaches fall significantly short in performance compared to fully-supervised learning ones. To bridge the performance gap without mask annotations, we propose a novel weakly-supervised framework that tackles RIS by decomposing it into three steps: obtaining instance masks for the object mentioned in the referencing instruction (segment), using zero-shot learning to select a potentially correct mask for the given instruction (select), and bootstrapping a model which allows for fixing the mistakes of zero-shot selection (correct). In our experiments, using only the first two steps (zero-shot segment and select) outperforms other zero-shot baselines by as much as 19%,
    
[^3]: 基于图谱的可解释年龄预测

    Atlas-Based Interpretable Age Prediction. (arXiv:2307.07439v1 [eess.IV])

    [http://arxiv.org/abs/2307.07439](http://arxiv.org/abs/2307.07439)

    本研究提出了一种基于图谱的可解释年龄预测方法，利用全身图像研究了各个身体部位的年龄相关变化。通过使用解释性方法和配准技术，确定了最能预测年龄的身体区域，并创下了整个身体年龄预测的最新水平。研究结果表明，脊柱、本原性背部肌肉和心脏区域是最重要的关注领域。

    

    年龄预测是医学评估和研究的重要部分，可以通过突出实际年龄和生物年龄之间的差异来帮助检测疾病和异常衰老。为了全面了解各个身体部位的年龄相关变化，我们使用了全身图像进行研究。我们利用Grad-CAM解释性方法确定最能预测一个人年龄的身体区域。通过使用配准技术生成整个人群的解释性图，我们将分析扩展到个体之外。此外，我们以一个平均绝对误差为2.76年的模型，创下了整个身体年龄预测的最新水平。我们的研究结果揭示了三个主要的关注领域：脊柱、本原性背部肌肉和心脏区域，其中心脏区域具有最重要的作用。

    Age prediction is an important part of medical assessments and research. It can aid in detecting diseases as well as abnormal ageing by highlighting the discrepancy between chronological and biological age. To gain a comprehensive understanding of age-related changes observed in various body parts, we investigate them on a larger scale by using whole-body images. We utilise the Grad-CAM interpretability method to determine the body areas most predictive of a person's age. We expand our analysis beyond individual subjects by employing registration techniques to generate population-wide interpretability maps. Furthermore, we set state-of-the-art whole-body age prediction with a model that achieves a mean absolute error of 2.76 years. Our findings reveal three primary areas of interest: the spine, the autochthonous back muscles, and the cardiac region, which exhibits the highest importance.
    
[^4]: MMBench: 您的多模态模型是全能球员吗？

    MMBench: Is Your Multi-modal Model an All-around Player?. (arXiv:2307.06281v1 [cs.CV])

    [http://arxiv.org/abs/2307.06281](http://arxiv.org/abs/2307.06281)

    MMBench是一个新型的多模态基准测试，旨在解决大型视觉语言模型评估的挑战，通过开发全面的评估流程和精心策划的数据集进行细粒度能力评估。

    

    最近，大型视觉语言模型在视觉信息的感知和推理能力方面取得了显著进展。然而，如何有效评估这些大型视觉语言模型仍然是一个主要障碍，阻碍了未来模型的发展。传统的基准测试，如VQAv2或COCO Caption提供了定量的性能测量，但在细粒度能力评估和非鲁棒评估指标方面存在不足。最近的主观基准测试，如OwlEval，通过整合人力资源，对模型的能力进行了全面评估，但不可扩展并且存在显著的偏见。针对这些挑战，我们提出了MMBench，一种新型的多模态基准测试。MMBench系统地开发了一个全面的评估流程，主要由两个元素组成。第一个元素是精心策划的数据集，在评估数量和多样性方面超越了现有的类似基准测试。

    Large vision-language models have recently achieved remarkable progress, exhibiting great perception and reasoning abilities concerning visual information. However, how to effectively evaluate these large vision-language models remains a major obstacle, hindering future model development. Traditional benchmarks like VQAv2 or COCO Caption provide quantitative performance measurements but suffer from a lack of fine-grained ability assessment and non-robust evaluation metrics. Recent subjective benchmarks, such as OwlEval, offer comprehensive evaluations of a model's abilities by incorporating human labor, but they are not scalable and display significant bias. In response to these challenges, we propose MMBench, a novel multi-modality benchmark. MMBench methodically develops a comprehensive evaluation pipeline, primarily comprised of two elements. The first element is a meticulously curated dataset that surpasses existing similar benchmarks in terms of the number and variety of evaluatio
    
[^5]: 高效的量化感知训练与自适应核心集选择

    Efficient Quantization-aware Training with Adaptive Coreset Selection. (arXiv:2306.07215v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.07215](http://arxiv.org/abs/2306.07215)

    本研究提出了一种用于改善量化感知训练的训练效率的方法，通过核心集选择和两个重要性指标来选择训练数据的子集。

    

    深度神经网络（DNN）的模型大小和计算量的增加，增加了对有效模型部署方法的需求。量化感知训练（QAT）是一种代表性的模型压缩方法，可以利用权重和激活中的冗余信息。然而，大多数现有的QAT方法需要在整个数据集上进行端到端训练，这会导致长时间的训练和高能耗。核心集选择是利用训练数据的冗余性提高数据效率的方法，在高效训练中被广泛应用。在这项工作中，我们提出了一种新的角度，通过核心集选择来提高量化感知训练的训练效率。基于QAT的特性，我们提出了两个指标：误差向量分数和不一致分数，用于量化训练过程中每个样本的重要性。基于这两个重要性指标，我们提出了一种量化感知的自适应核心集选择（ACS）方法，用于选择训练数据的子集。

    The expanding model size and computation of deep neural networks (DNNs) have increased the demand for efficient model deployment methods. Quantization-aware training (QAT) is a representative model compression method to leverage redundancy in weights and activations. However, most existing QAT methods require end-to-end training on the entire dataset, which suffers from long training time and high energy costs. Coreset selection, aiming to improve data efficiency utilizing the redundancy of training data, has also been widely used for efficient training. In this work, we propose a new angle through the coreset selection to improve the training efficiency of quantization-aware training. Based on the characteristics of QAT, we propose two metrics: error vector score and disagreement score, to quantify the importance of each sample during training. Guided by these two metrics of importance, we proposed a quantization-aware adaptive coreset selection (ACS) method to select the data for the
    
[^6]: DiracDiffusion: 确保数据一致性的去噪和增量重建

    DiracDiffusion: Denoising and Incremental Reconstruction with Assured Data-Consistency. (arXiv:2303.14353v1 [eess.IV])

    [http://arxiv.org/abs/2303.14353](http://arxiv.org/abs/2303.14353)

    DiracDiffusion是一种新的逆问题求解框架，可以应用于图像去噪和增量重建，并保证数据一致性。

    

    扩散模型在许多计算机视觉任务中（包括图像恢复）已经建立了新的技术水平。基于扩散的逆问题求解器从严重损坏的测量数据中生成出具有出色视觉质量的重建结果。然而，在所谓的感知-失真权衡中，感知效果优秀的重建结果通常是以退化的失真度量（如PSNR）为代价的。失真度量衡量对观察的忠实度，这在逆问题中是一个至关重要的要求。在这项工作中，我们提出了一种新的逆问题求解框架，即我们假设观察值来自一个随机劣化过程，逐渐降低和噪声化原始干净图像，然后学习逆转劣化过程以恢复干净图像。我们的技术在整个逆转过程中保持与原始测量的一致性，并允许在感知质量和数据一致性之间取得巨大的灵活性。我们的方法称为DiracDiffusion，因为它基于由Dirac能量函数引导的扩散过程。我们在包括去噪和增量重建在内的几个具有挑战性的图像恢复任务中展示了我们方法的有效性。

    Diffusion models have established new state of the art in a multitude of computer vision tasks, including image restoration. Diffusion-based inverse problem solvers generate reconstructions of exceptional visual quality from heavily corrupted measurements. However, in what is widely known as the perception-distortion trade-off, the price of perceptually appealing reconstructions is often paid in declined distortion metrics, such as PSNR. Distortion metrics measure faithfulness to the observation, a crucial requirement in inverse problems. In this work, we propose a novel framework for inverse problem solving, namely we assume that the observation comes from a stochastic degradation process that gradually degrades and noises the original clean image. We learn to reverse the degradation process in order to recover the clean image. Our technique maintains consistency with the original measurement throughout the reverse process, and allows for great flexibility in trading off perceptual qu
    

