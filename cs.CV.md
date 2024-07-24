# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation](https://arxiv.org/abs/2404.00636) | 本文提出了一种一次性的3D感知肖像动画方法Export3D，通过引入三平面生成器和对比预训练框架，实现了控制给定肖像图像的面部表情和摄像机视角，提供了一种新的表达方式。 |
| [^2] | [DetToolChain: A New Prompting Paradigm to Unleash Detection Ability of MLLM](https://arxiv.org/abs/2403.12488) | DetToolChain提出了一种新的提示范式，可以释放MLLM的零拍摄物体检测能力，并通过检测链式思维自动化任务分解和逐步框细化规划。 |
| [^3] | [Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) | 评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标 |
| [^4] | [How Easy is It to Fool Your Multimodal LLMs? An Empirical Analysis on Deceptive Prompts](https://arxiv.org/abs/2402.13220) | 对多模态LLMs的欺骗性提示进行了实证分析，提出包含850个测试样本的基准测试MAD-Bench，发现GPT-4V在该基准测试上准确率较高，而其他模型性能差距显著。 |
| [^5] | [RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning](https://arxiv.org/abs/2402.08823) | RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。 |
| [^6] | [Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting.](http://arxiv.org/abs/2401.15318) | 高斯喷溅技术相结合物理基础动画和3D高斯喷溅，可以在虚拟场景中创造出无可比拟的效果，同时实现渲染、视图合成以及固体和流体的动态管理和交互。 |
| [^7] | [Tweets to Citations: Unveiling the Impact of Social Media Influencers on AI Research Visibility.](http://arxiv.org/abs/2401.13782) | 本文研究了社交媒体影响者在提高机器学习研究的可见性方面的作用，发现被这些影响者认可的论文引用次数显著增加，中位数引用次数比对照组高2-3倍。此外，该研究还探讨了被展示作者的地理、性别和机构多样性。 |
| [^8] | [Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering.](http://arxiv.org/abs/2401.10711) | 本论文提出了一种使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题的方法。通过将问题和答案对作为事件描述，找到多个关键帧作为目标时刻，并利用这些时刻作为伪标签来强制LMMs进行推理。所提出的方法使用轻量级的基于高斯的对比基础模块（GCG）来学习时效结构。 |
| [^9] | [ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy.](http://arxiv.org/abs/2311.09215) | ConvNet和Transformer架构在监督和CLIP训练下，超越了ImageNet准确率的对比分析中发现它们在错误类型、输出校准、可转移性和特征的不变性等方面存在差异，突出了需要更加细致分析的必要性。 |
| [^10] | [Defending Our Privacy With Backdoors.](http://arxiv.org/abs/2310.08320) | 本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。 |
| [^11] | [Unsupervised anomaly localization in high-resolution breast scans using deep pluralistic image completion.](http://arxiv.org/abs/2305.03098) | 本文提出了一种利用深度多元图像完成方法进行乳腺扫描异常定位的无监督方法，该方法通过探索完成的多元性来提高评估标准的精度。 |
| [^12] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |
| [^13] | [Laplacian Segmentation Networks: Improved Epistemic Uncertainty from Spatial Aleatoric Uncertainty.](http://arxiv.org/abs/2303.13123) | 所提出的拉普拉斯分割网络可同时捕获图像分割中的认知和随机不确定性，成功将高认知不确定性分配到OOF目标中。 |
| [^14] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 学习生成条件化三平面用于3D感知表情可控肖像动画

    Learning to Generate Conditional Tri-plane for 3D-aware Expression Controllable Portrait Animation

    [https://arxiv.org/abs/2404.00636](https://arxiv.org/abs/2404.00636)

    本文提出了一种一次性的3D感知肖像动画方法Export3D，通过引入三平面生成器和对比预训练框架，实现了控制给定肖像图像的面部表情和摄像机视角，提供了一种新的表达方式。

    

    在本文中，我们提出了一种一次性的3D感知肖像动画方法Export3D，能够控制给定肖像图像的面部表情和摄像机视角。为了实现这一目标，我们引入了一个三平面生成器，通过将3DMM的表情参数转移到源图像中直接生成3D先验的三平面。然后，通过可微分体积渲染将三平面解码为不同视角的图像。现有的肖像动画方法严重依赖于图像变形来在运动空间中传输表情，挑战在外观和表情的分离上。相比之下，我们提出了一个用于无外观表情参数的对比预训练框架，消除了在传输跨身份表达时不良外观交换。大量实验证明，我们的预训练框架能够学习隐藏在3DMM中的无外观表达表示。

    arXiv:2404.00636v1 Announce Type: cross  Abstract: In this paper, we present Export3D, a one-shot 3D-aware portrait animation method that is able to control the facial expression and camera view of a given portrait image. To achieve this, we introduce a tri-plane generator that directly generates a tri-plane of 3D prior by transferring the expression parameter of 3DMM into the source image. The tri-plane is then decoded into the image of different view through a differentiable volume rendering. Existing portrait animation methods heavily rely on image warping to transfer the expression in the motion space, challenging on disentanglement of appearance and expression. In contrast, we propose a contrastive pre-training framework for appearance-free expression parameter, eliminating undesirable appearance swap when transferring a cross-identity expression. Extensive experiments show that our pre-training framework can learn the appearance-free expression representation hidden in 3DMM, and 
    
[^2]: DetToolChain：一种释放MLLM检测能力的新提示范式

    DetToolChain: A New Prompting Paradigm to Unleash Detection Ability of MLLM

    [https://arxiv.org/abs/2403.12488](https://arxiv.org/abs/2403.12488)

    DetToolChain提出了一种新的提示范式，可以释放MLLM的零拍摄物体检测能力，并通过检测链式思维自动化任务分解和逐步框细化规划。

    

    我们提出DetToolChain，一种新颖的提示范式，用于释放多模态大语言模型（MLLMs）的零拍摄物体检测能力，如GPT-4V和Gemini。我们的方法包括一个受高精度检测先验启发的检测提示工具包和一个实现这些提示的新Chain-of-Thought。具体来说，工具包中的提示旨在引导MLLM集中在区域信息上（例如，放大），按照测量标准阅读坐标（例如，叠加标尺和指南针），并从上下文信息中推断（例如，叠加场景图）。基于这些工具，新的检测Chain-of-Thought可以自动将任务分解为简单的子任务，诊断预测，并规划逐步框细化。我们的框架的有效性在一系列检测任务中得到了证实，特别是在困难情况下。与现有的最先进技术相比

    arXiv:2403.12488v1 Announce Type: cross  Abstract: We present DetToolChain, a novel prompting paradigm, to unleash the zero-shot object detection ability of multimodal large language models (MLLMs), such as GPT-4V and Gemini. Our approach consists of a detection prompting toolkit inspired by high-precision detection priors and a new Chain-of-Thought to implement these prompts. Specifically, the prompts in the toolkit are designed to guide the MLLM to focus on regional information (e.g., zooming in), read coordinates according to measure standards (e.g., overlaying rulers and compasses), and infer from the contextual information (e.g., overlaying scene graphs). Building upon these tools, the new detection chain-of-thought can automatically decompose the task into simple subtasks, diagnose the predictions, and plan for progressive box refinements. The effectiveness of our framework is demonstrated across a spectrum of detection tasks, especially hard cases. Compared to existing state-of-
    
[^3]: 评估文本到图像合成：图像质量度量的调查与分类

    Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics

    [https://arxiv.org/abs/2403.11821](https://arxiv.org/abs/2403.11821)

    评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标

    

    最近，通过利用语言和视觉结合的基础模型，推动了文本到图像合成方面的进展。这些模型在互联网或其他大规模数据库中的海量文本-图像对上进行了预训练。随着对高质量图像生成的需求转向确保文本与图像之间的内容对齐，已开发了新颖的评估度量标准，旨在模拟人类判断。因此，研究人员开始收集具有越来越复杂注释的数据集，以研究视觉语言模型的组成性及其作为文本与图像内容组成对齐质量度量的其纳入。在这项工作中，我们全面介绍了现有的文本到图像评估指标，并提出了一个新的分类法来对这些指标进行分类。我们还审查了经常采用的文本-图像基准数据集

    arXiv:2403.11821v1 Announce Type: cross  Abstract: Recent advances in text-to-image synthesis have been enabled by exploiting a combination of language and vision through foundation models. These models are pre-trained on tremendous amounts of text-image pairs sourced from the World Wide Web or other large-scale databases. As the demand for high-quality image generation shifts towards ensuring content alignment between text and image, novel evaluation metrics have been developed with the aim of mimicking human judgments. Thus, researchers have started to collect datasets with increasingly complex annotations to study the compositionality of vision-language models and their incorporation as a quality measure of compositional alignment between text and image contents. In this work, we provide a comprehensive overview of existing text-to-image evaluation metrics and propose a new taxonomy for categorizing these metrics. We also review frequently adopted text-image benchmark datasets befor
    
[^4]: 有多容易欺骗多模态LLMs？关于欺骗性提示的实证分析

    How Easy is It to Fool Your Multimodal LLMs? An Empirical Analysis on Deceptive Prompts

    [https://arxiv.org/abs/2402.13220](https://arxiv.org/abs/2402.13220)

    对多模态LLMs的欺骗性提示进行了实证分析，提出包含850个测试样本的基准测试MAD-Bench，发现GPT-4V在该基准测试上准确率较高，而其他模型性能差距显著。

    

    多模态大型语言模型（MLLMs）的显著进展并没有使它们免疫各种挑战，特别是在处理带有欺骗性信息的提示时，会产生幻觉般的回应。为了定量评估这种脆弱性，我们提出了MAD-Bench，一个精心策划的基准测试，包含850个测试样本，分为6个类别，如不存在的对象、对象数量、空间关系和视觉混淆。我们对流行的MLLMs进行了全面分析，包括GPT-4V、Gemini-Pro，以及开源模型，如LLaVA-1.5和CogVLM。实证研究中，我们观察到GPT-4V和其他模型之间存在着显著的性能差距；之前的鲁棒指令调整模型，如LRV-Instruction和LLaVA-RLHF，在这个新基准测试中并不有效。虽然GPT-4V在MAD-Bench上取得了75.02%的准确率，但其他任何模型在我们的实验中都没有达到这一水平。

    arXiv:2402.13220v1 Announce Type: cross  Abstract: The remarkable advancements in Multimodal Large Language Models (MLLMs) have not rendered them immune to challenges, particularly in the context of handling deceptive information in prompts, thus producing hallucinated responses under such conditions. To quantitatively assess this vulnerability, we present MAD-Bench, a carefully curated benchmark that contains 850 test samples divided into 6 categories, such as non-existent objects, count of objects, spatial relationship, and visual confusion. We provide a comprehensive analysis of popular MLLMs, ranging from GPT-4V, Gemini-Pro, to open-sourced models, such as LLaVA-1.5 and CogVLM. Empirically, we observe significant performance gaps between GPT-4V and other models; and previous robust instruction-tuned models, such as LRV-Instruction and LLaVA-RLHF, are not effective on this new benchmark. While GPT-4V achieves 75.02% accuracy on MAD-Bench, the accuracy of any other model in our exper
    
[^5]: RanDumb: 一种质疑持续表示学习效果的简单方法

    RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning

    [https://arxiv.org/abs/2402.08823](https://arxiv.org/abs/2402.08823)

    RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。

    

    我们提出了RanDumb来检验持续表示学习的效果。RanDumb将原始像素使用一个固定的随机变换嵌入，这个变换近似了RBF-Kernel，在看到任何数据之前初始化，并学习一个简单的线性分类器。我们提出了一个令人惊讶且一致的发现：在众多持续学习基准测试中，RanDumb在性能上明显优于使用深度网络进行持续学习的表示学习，这表明在这些情景下表示学习的性能较差。RanDumb不存储样本，并在数据上进行单次遍历，一次处理一个样本。它与GDumb相辅相成，在GDumb性能特别差的低样本情况下运行。当将RanDumb扩展到使用预训练模型替换随机变换的情景时，我们得出相同一致的结论。我们的调查结果既令人惊讶又令人担忧，因为表示学习在这些情况下表现糟糕。

    arXiv:2402.08823v1 Announce Type: cross Abstract: We propose RanDumb to examine the efficacy of continual representation learning. RanDumb embeds raw pixels using a fixed random transform which approximates an RBF-Kernel, initialized before seeing any data, and learns a simple linear classifier on top. We present a surprising and consistent finding: RanDumb significantly outperforms the continually learned representations using deep networks across numerous continual learning benchmarks, demonstrating the poor performance of representation learning in these scenarios. RanDumb stores no exemplars and performs a single pass over the data, processing one sample at a time. It complements GDumb, operating in a low-exemplar regime where GDumb has especially poor performance. We reach the same consistent conclusions when RanDumb is extended to scenarios with pretrained models replacing the random transform with pretrained feature extractor. Our investigation is both surprising and alarming as
    
[^6]: 高斯喷溅：利用高斯飘落动态合成流体

    Gaussian Splashing: Dynamic Fluid Synthesis with Gaussian Splatting. (arXiv:2401.15318v1 [cs.GR])

    [http://arxiv.org/abs/2401.15318](http://arxiv.org/abs/2401.15318)

    高斯喷溅技术相结合物理基础动画和3D高斯喷溅，可以在虚拟场景中创造出无可比拟的效果，同时实现渲染、视图合成以及固体和流体的动态管理和交互。

    

    我们展示了将物理基础动画与3D高斯喷溅（3DGS）相结合的可行性，以在使用3DGS重建的虚拟场景中创建新效果。利用高斯喷溅和基于位置的动力学（PBD）在底层表示中的一致性，我们以连贯的方式管理渲染、视图合成以及固体和流体的动态。类似于高斯着色器，我们通过添加法线增强每个高斯核，将核的方向与表面法线对齐，以改进PBD模拟。这种方法有效消除了固体旋转变形产生的尖峰噪声。它还使我们能够将基于物理的渲染集成到流体的动态表面反射中。因此，我们的框架能够真实地复现动态流体上的表面亮点，并促进场景对象与流体之间的交互。

    We demonstrate the feasibility of integrating physics-based animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes reconstructed using 3DGS. Leveraging the coherence of the Gaussian splatting and position-based dynamics (PBD) in the underlying representation, we manage rendering, view synthesis, and the dynamics of solids and fluids in a cohesive manner. Similar to Gaussian shader, we enhance each Gaussian kernel with an added normal, aligning the kernel's orientation with the surface normal to refine the PBD simulation. This approach effectively eliminates spiky noises that arise from rotational deformation in solids. It also allows us to integrate physically based rendering to augment the dynamic surface reflections on fluids. Consequently, our framework is capable of realistically reproducing surface highlights on dynamic fluids and facilitating interactions between scene objects and fluids from new views. For more information, pl
    
[^7]: 从推特到引用：揭示社交媒体影响者对人工智能研究可见性的影响

    Tweets to Citations: Unveiling the Impact of Social Media Influencers on AI Research Visibility. (arXiv:2401.13782v1 [cs.DL])

    [http://arxiv.org/abs/2401.13782](http://arxiv.org/abs/2401.13782)

    本文研究了社交媒体影响者在提高机器学习研究的可见性方面的作用，发现被这些影响者认可的论文引用次数显著增加，中位数引用次数比对照组高2-3倍。此外，该研究还探讨了被展示作者的地理、性别和机构多样性。

    

    随着人工智能和机器学习会议上被接受的论文数量达到数千篇，研究人员如何获取和阅读研究论文变得不清楚。本文研究了社交媒体影响者在增强机器学习研究可见性中的作用，特别是他们分享的论文引用次数。我们编制了一个包括8000多篇论文的全面数据集，涵盖了2018年12月至2023年10月的推特，以及基于出版年份、会议地点和摘要主题进行1：1匹配的对照组。我们的分析揭示了这些影响者认可的论文引用次数显著增加，中位数引用次数比对照组高2-3倍。此外，该研究还深入研究了被展示作者的地理、性别和机构多样性。这些发现突显了社交媒体在学术交流中的不断扩大的影响力，并强调了当今数字化时代不断发展的生态系统的重要性。

    As the number of accepted papers at AI and ML conferences reaches into the thousands, it has become unclear how researchers access and read research publications. In this paper, we investigate the role of social media influencers in enhancing the visibility of machine learning research, particularly the citation counts of papers they share. We have compiled a comprehensive dataset of over 8,000 papers, spanning tweets from December 2018 to October 2023, alongside 1:1 matched controls based on publication year, venue, and abstract topics. Our analysis reveals a significant increase in citations for papers endorsed by these influencers, with median citation counts 2-3 times higher than those of the control group. Additionally, the study delves into the geographic, gender, and institutional diversity of highlighted authors. These findings highlight the expanding influence of social media in scholarly communication and underscore the importance of an evolving ecosystem in today's digital a
    
[^8]: 使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题

    Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering. (arXiv:2401.10711v1 [cs.CV])

    [http://arxiv.org/abs/2401.10711](http://arxiv.org/abs/2401.10711)

    本论文提出了一种使用大型多模型的弱监督高斯对比基础模型来处理视频问答问题的方法。通过将问题和答案对作为事件描述，找到多个关键帧作为目标时刻，并利用这些时刻作为伪标签来强制LMMs进行推理。所提出的方法使用轻量级的基于高斯的对比基础模块（GCG）来学习时效结构。

    

    视频问答（VideoQA）旨在基于观察到的视频信息回答自然语言问题。尽管大型多模型（LMMs）在图像语言理解和推理方面取得了近期的成功，但它们在处理视频问答方面还不足够，仅仅是将均匀采样的帧作为视觉输入，忽略了与问题相关的视觉线索。此外，现有的视频问答数据集中没有针对问题关键时间戳的人工注释。基于此，我们提出了一种新的弱监督框架，强制LMMs使用问题关键时刻作为视觉输入推理出答案。具体来说，我们将问题和答案对合并为事件描述，以找到多个关键帧作为目标时刻，这些时刻将作为伪标签。通过将这些伪标签作为额外的弱监督，我们设计了一个轻量级的基于高斯的对比基础模块（GCG）。GCG学习多个高斯函数来描述时效结构。

    Video Question Answering (VideoQA) aims to answer natural language questions based on the information observed in videos. Despite the recent success of Large Multimodal Models (LMMs) in image-language understanding and reasoning, they deal with VideoQA insufficiently by simply taking uniformly sampled frames as visual inputs, which ignores question-relevant visual clues. Moreover, there are no human annotations for question-critical timestamps in existing VideoQA datasets. In light of this, we propose a novel weakly supervised framework to enforce the LMMs to reason out the answers with question-critical moments as visual inputs. Specifically, we fuse the question and answer pairs as event descriptions to find multiple keyframes as target moments, which will be pseudo-labels. With these pseudo-labels as additionally weak supervision, we devise a lightweight Gaussian-based Contrastive Grounding (GCG) module. GCG learns multiple Gaussian functions to characterize the temporal structure o
    
[^9]: ConvNet vs Transformer, Supervised vs CLIP: 超越ImageNet准确率

    ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy. (arXiv:2311.09215v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.09215](http://arxiv.org/abs/2311.09215)

    ConvNet和Transformer架构在监督和CLIP训练下，超越了ImageNet准确率的对比分析中发现它们在错误类型、输出校准、可转移性和特征的不变性等方面存在差异，突出了需要更加细致分析的必要性。

    

    现代计算机视觉为实践者提供了多种模型选择，对于特定应用从多个选项中选择模型可能具有挑战性。传统上，通过它们在ImageNet上的分类准确率来比较竞争模型架构和训练协议。然而，这个单一指标无法完全捕捉到对于专业任务至关重要的性能细微差别。在这项工作中，我们对ConvNet和Vision Transformer架构在监督和CLIP训练范式下进行了深入的比较分析，超越了ImageNet的准确率。尽管我们选择的模型在ImageNet准确率和计算需求上相似，我们发现它们在许多其他方面存在差异：错误类型、输出校准、可转移性和特征的不变性等。传统指标无法捕捉到的这种模型特性差异，突出了在选择不同模型时需要更加细致分析的必要性。

    Modern computer vision offers a great variety of models to practitioners, and selecting a model from multiple options for specific applications can be challenging. Conventionally, competing model architectures and training protocols are compared by their classification accuracy on ImageNet. However, this single metric does not fully capture performance nuances critical for specialized tasks. In this work, we conduct an in-depth comparative analysis of model behaviors beyond ImageNet accuracy, for both ConvNet and Vision Transformer architectures, each across supervised and CLIP training paradigms. Although our selected models have similar ImageNet accuracies and compute requirements, we find that they differ in many other aspects: types of mistakes, output calibration, transferability, and feature invariance, among others. This diversity in model characteristics, not captured by traditional metrics, highlights the need for more nuanced analysis when choosing among different models. Our
    
[^10]: 使用后门技术保护我们的隐私

    Defending Our Privacy With Backdoors. (arXiv:2310.08320v1 [cs.LG])

    [http://arxiv.org/abs/2310.08320](http://arxiv.org/abs/2310.08320)

    本研究提出了一种基于后门攻击的防御方法，通过对模型进行策略性插入后门，对齐敏感短语与中性术语的嵌入，以删除训练数据中的私人信息。实证结果显示该方法的有效性。

    

    在使用未经筛选、常常包含敏感信息的网页数据训练大型人工智能模型的情况下，隐私问题成为了一个重要的关注点。其中一个问题是，攻击者可以利用隐私攻击的方法提取出训练数据的信息。然而，如何在不降低模型性能的情况下去除特定信息是一个不容易解决且具有挑战性的问题。我们提出了一个基于后门攻击的简单而有效的防御方法，用于从模型中删除私人信息，如个人姓名，特别是针对文本编码器的。具体而言，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语的嵌入对齐，例如用"a person"代替人名。我们的实证结果通过对零样本分类器使用专门的隐私攻击测试表明了我们基于后门的防御方法的效果。我们的方法提供了一个新的"双重用途"的视角。

    The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspecti
    
[^11]: 使用深度多元图像完成进行高分辨率乳腺扫描的无监督异常定位

    Unsupervised anomaly localization in high-resolution breast scans using deep pluralistic image completion. (arXiv:2305.03098v1 [eess.IV])

    [http://arxiv.org/abs/2305.03098](http://arxiv.org/abs/2305.03098)

    本文提出了一种利用深度多元图像完成方法进行乳腺扫描异常定位的无监督方法，该方法通过探索完成的多元性来提高评估标准的精度。

    

    数字乳腺摄影中的自动肿瘤检测是一项困难的任务，由于肿瘤很少出现，乳房组织变异和高分辨率。针对这个问题，鉴于异常图像的稀缺性和正常图像的丰富性，异常检测/定位方法可能非常适合。然而，机器学习中大部分异常定位研究集中在非医学数据集上，我们发现这些方法在医学图像数据集上适应性不足。这个问题可以通过解决图像完成视角下的任务得到缓解，其中异常的存在可以通过原始外观与其环境条件下自动完成之间的差异来指示。然而，在DBT数据集中，往往有很多相同环境条件下的有效的正常完成，使这个评估标准不太精确。为了解决这个问题，我们考虑通过探索完成的多元性来进行图像完成。

    Automated tumor detection in Digital Breast Tomosynthesis (DBT) is a difficult task due to natural tumor rarity, breast tissue variability, and high resolution. Given the scarcity of abnormal images and the abundance of normal images for this problem, an anomaly detection/localization approach could be well-suited. However, most anomaly localization research in machine learning focuses on non-medical datasets, and we find that these methods fall short when adapted to medical imaging datasets. The problem is alleviated when we solve the task from the image completion perspective, in which the presence of anomalies can be indicated by a discrepancy between the original appearance and its auto-completion conditioned on the surroundings. However, there are often many valid normal completions given the same surroundings, especially in the DBT dataset, making this evaluation criterion less precise. To address such an issue, we consider pluralistic image completion by exploring the distributi
    
[^12]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    
[^13]: 拉普拉斯分割网络: 从空间数据不确定性到改进的认知不确定性

    Laplacian Segmentation Networks: Improved Epistemic Uncertainty from Spatial Aleatoric Uncertainty. (arXiv:2303.13123v1 [cs.CV])

    [http://arxiv.org/abs/2303.13123](http://arxiv.org/abs/2303.13123)

    所提出的拉普拉斯分割网络可同时捕获图像分割中的认知和随机不确定性，成功将高认知不确定性分配到OOF目标中。

    

    媒体图像常常会出现非正常的情况，例如因为位置或扫描器的不同或图像损坏等原因而经常出现，这种媒体图像可能会对下游临床诊断或治疗产生影响。为了确保对这种错误分割的鲁棒性，我们提出了拉普拉斯分割网络（LSN），其能够共同建模图像分割中的认知（模型）不确定性和空间数据（随机）不确定性。我们使用具有空间相关性的logit分布捕获数据的不确定性。对于模型不确定性，我们提出了针对高维输出和具有跳过连接的大型神经网络的第一个拉普拉斯权重后验的逼近。从实证上，我们证明了建模空间像素相关性使得拉普拉斯分割网络能够将高认知不确定性成功分配到图像中的OOF目标。

    Out of distribution (OOD) medical images are frequently encountered, e.g. because of site- or scanner differences, or image corruption. OOD images come with a risk of incorrect image segmentation, potentially negatively affecting downstream diagnoses or treatment. To ensure robustness to such incorrect segmentations, we propose Laplacian Segmentation Networks (LSN) that jointly model epistemic (model) and aleatoric (data) uncertainty in image segmentation. We capture data uncertainty with a spatially correlated logit distribution. For model uncertainty, we propose the first Laplace approximation of the weight posterior that scales to large neural networks with skip connections that have high-dimensional outputs. Empirically, we demonstrate that modelling spatial pixel correlation allows the Laplacian Segmentation Network to successfully assign high epistemic uncertainty to out-of-distribution objects appearing within images.
    
[^14]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

