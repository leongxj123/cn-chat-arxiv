# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Listening to the Noise: Blind Denoising with Gibbs Diffusion](https://arxiv.org/abs/2402.19455) | 引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。 |
| [^2] | [Wavelet Scattering Transform for Bioacustics: Application to Watkins Marine Mammal Sound Database](https://arxiv.org/abs/2402.17775) | 本研究提出了在Watkins海洋哺乳动物声音数据库上应用Wavelet散射变换（WST）和Mel频谱图预处理的方法，在分类任务中取得了较高的准确率。 |
| [^3] | [Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment](https://arxiv.org/abs/2402.13561) | 该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。 |
| [^4] | [SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models](https://arxiv.org/abs/2402.05935) | 本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。 |
| [^5] | [DeCoF: Generated Video Detection via Frame Consistency](https://arxiv.org/abs/2402.02085) | 通过帧一致性原则，DeCoF是一个简单但有效的生成视频检测模型，可以消除空间伪影的影响，并表现出强大的泛化能力。 |
| [^6] | [Deep Learning for Multi-Label Learning: A Comprehensive Survey.](http://arxiv.org/abs/2401.16549) | 深度学习在多标签学习中的综合调研，旨在审视深度学习在解决多标签分类中的挑战方面的最新进展。 |
| [^7] | [ODIN: A Single Model for 2D and 3D Perception.](http://arxiv.org/abs/2401.02416) | ODIN是一个模型，可以同时对2D RGB图像和3D点云进行分割和标记，使用变压器架构进行2D和3D视图间的信息融合。 |
| [^8] | [Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer.](http://arxiv.org/abs/2310.02674) | 本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，提出了一种基于对象引导的Transformer架构，从而拓宽了变化检测任务的范围，并显著减少了计算开销和内存负担。 |
| [^9] | [Single-Model Attribution via Final-Layer Inversion.](http://arxiv.org/abs/2306.06210) | 本文提出了一种利用最终层反演和异常检测的开放式单模型归因方法，解决了以往方法要么局限于封闭式环境、要么需要对生成模型进行不必要的改变的问题。实验结果表明该方法优于现有方法。 |
| [^10] | [Neuromorphic Visual Scene Understanding with Resonator Networks.](http://arxiv.org/abs/2208.12880) | 本论文提出了一种基于神经形态的解决方案，利用高效的因式分解网络来理解视觉场景并推断物体和姿势。关键创新包括基于复值向量的计算框架VSA、用于处理平移和旋转的分层谐振器网络HRN设计，以及在神经形态硬件上实现复值谐振器网络的多组分脉冲相位神经元模型。 |

# 详细

[^1]: 听噪声：使用Gibbs扩散进行盲去噪

    Listening to the Noise: Blind Denoising with Gibbs Diffusion

    [https://arxiv.org/abs/2402.19455](https://arxiv.org/abs/2402.19455)

    引入了Gibbs扩散（GDiff）方法，通过交替采样信号先验和噪声分布族，以及蒙特卡洛采样来推断噪声参数，解决了盲去噪中需要知道噪声水平和协方差的问题。

    

    近年来，去噪问题与深度生成模型的发展密不可分。特别是，扩散模型被训练成去噪器，它们所建模的分布与贝叶斯图像中的去噪先验相符。然而，通过基于扩散的后验采样进行去噪需要知道噪声水平和协方差，这阻碍了盲去噪。我们通过引入 Gibbs扩散（GDiff）克服了这一限制，这是一种通用方法论，可以处理信号和噪声参数的后验采样。假设任意参数化的高斯噪声，我们开发了一种Gibbs算法，交替地从条件扩散模型中进行采样，该模型经过训练将信号先验映射到噪声分布族，以及一个蒙特卡洛采样器来推断噪声参数。我们的理论分析突出了潜在的缺陷，指导了诊断的使用，并量化了Gibbs s中的错误。

    arXiv:2402.19455v1 Announce Type: cross  Abstract: In recent years, denoising problems have become intertwined with the development of deep generative models. In particular, diffusion models are trained like denoisers, and the distribution they model coincide with denoising priors in the Bayesian picture. However, denoising through diffusion-based posterior sampling requires the noise level and covariance to be known, preventing blind denoising. We overcome this limitation by introducing Gibbs Diffusion (GDiff), a general methodology addressing posterior sampling of both the signal and the noise parameters. Assuming arbitrary parametric Gaussian noise, we develop a Gibbs algorithm that alternates sampling steps from a conditional diffusion model trained to map the signal prior to the family of noise distributions, and a Monte Carlo sampler to infer the noise parameters. Our theoretical analysis highlights potential pitfalls, guides diagnostic usage, and quantifies errors in the Gibbs s
    
[^2]: Wavelet散射变换在生物声学中的应用：以Watkins海洋哺乳动物声音数据库为例

    Wavelet Scattering Transform for Bioacustics: Application to Watkins Marine Mammal Sound Database

    [https://arxiv.org/abs/2402.17775](https://arxiv.org/abs/2402.17775)

    本研究提出了在Watkins海洋哺乳动物声音数据库上应用Wavelet散射变换（WST）和Mel频谱图预处理的方法，在分类任务中取得了较高的准确率。

    

    海洋哺乳动物的交流是一个复杂的领域，受到鸣叫的多样性和环境因素的影响。Watkins海洋哺乳动物声音数据库（WMMD）是一个广泛应用于机器学习中的标记数据集。本研究首先重点介绍了该数据集上最新的基准记录，着重澄清数据准备和预处理方法。随后，我们提出了在STFT基础上应用Wavelet散射变换（WST）的方法。研究还探讨了使用自适应深层架构和残差层进行分类任务。我们在准确率上使用WST比现有分类架构提高了6％，使用Mel频谱图预处理提高了8％，从而有效地减少了

    arXiv:2402.17775v1 Announce Type: cross  Abstract: Marine mammal communication is a complex field, hindered by the diversity of vocalizations and environmental factors. The Watkins Marine Mammal Sound Database (WMMD) is an extensive labeled dataset used in machine learning applications. However, the methods for data preparation, preprocessing, and classification found in the literature are quite disparate. This study first focuses on a brief review of the state-of-the-art benchmarks on the dataset, with an emphasis on clarifying data preparation and preprocessing methods. Subsequently, we propose the application of the Wavelet Scattering Transform (WST) in place of standard methods based on the Short-Time Fourier Transform (STFT). The study also tackles a classification task using an ad-hoc deep architecture with residual layers. We outperform the existing classification architecture by $6\%$ in accuracy using WST and $8\%$ using Mel spectrogram preprocessing, effectively reducing by h
    
[^3]: 认知视觉语言映射器：通过增强视觉知识对齐推进多模态理解

    Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

    [https://arxiv.org/abs/2402.13561](https://arxiv.org/abs/2402.13561)

    该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。

    

    评估和反思当前大型多模态模型（LMMs）的现状，我们观察到广泛使用的视觉语言投影方法（如Q-former或MLP）侧重于图像-文本描述的对齐，但忽略了视觉知识维度的对齐，即将视觉与其相关知识连接起来。视觉知识在分析、推断和解释视觉信息方面起着重要作用，有助于提高基于知识的视觉问题答案的准确性。本文主要探讨通过视觉语言知识对齐来改进LMMs，特别针对挑战知识型视觉问答（VQA）。为此，我们提出了一个认知视觉语言映射器（CVLM），其中包含一个预训练的视觉知识对齐器（VKA）和一个用于多模态指令调节阶段的细粒度知识适配器（FKA）。具体来说，我们基于

    arXiv:2402.13561v1 Announce Type: new  Abstract: Evaluating and Rethinking the current landscape of Large Multimodal Models (LMMs), we observe that widely-used visual-language projection approaches (e.g., Q-former or MLP) focus on the alignment of image-text descriptions yet ignore the visual knowledge-dimension alignment, i.e., connecting visuals to their relevant knowledge. Visual knowledge plays a significant role in analyzing, inferring, and interpreting information from visuals, helping improve the accuracy of answers to knowledge-based visual questions. In this paper, we mainly explore improving LMMs with visual-language knowledge alignment, especially aimed at challenging knowledge-based visual question answering (VQA). To this end, we present a Cognitive Visual-Language Mapper (CVLM), which contains a pretrained Visual Knowledge Aligner (VKA) and a Fine-grained Knowledge Adapter (FKA) used in the multimodal instruction tuning stage. Specifically, we design the VKA based on the 
    
[^4]: SPHINX-X: 扩展数据和参数用于一系列多模态大型语言模型

    SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models

    [https://arxiv.org/abs/2402.05935](https://arxiv.org/abs/2402.05935)

    本论文介绍了SPHINX-X，一种扩展的多模态大型语言模型系列。通过改进架构和训练效率，我们成功构建了一系列参数大小和多语言能力不同的MLLMs，与数据和参数规模有强相关性。

    

    我们提出SPHINX-X，一种基于SPHINX开发的广泛多模态大型语言模型（MLLM）系列。为了改善架构和训练效率，我们通过移除冗余的视觉编码器、绕过完全填充的子图像，并将多阶段训练简化成为一阶段的全集合模式，修改了SPHINX框架。为了充分发挥MLLM的潜力，我们组装了一个综合的跨语言、跨视觉和视觉-语言任务的多领域、多模态的数据集，涵盖了公开可用的资源。我们进一步使用我们的OCR密集和Mark数据集丰富这个收集，扩展了多样性和普适性。通过对不同基础LLM进行训练，包括TinyLlama1.1B、InternLM2-7B、LLaMA2-13B和Mixtral8x7B，我们获得了一系列参数大小和多语言能力变化的MLLMs。全面的基准测试揭示了多模态性能与数据和参数规模之间的强相关性。

    We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. 
    
[^5]: DeCoF:通过帧一致性进行生成视频检测

    DeCoF: Generated Video Detection via Frame Consistency

    [https://arxiv.org/abs/2402.02085](https://arxiv.org/abs/2402.02085)

    通过帧一致性原则，DeCoF是一个简单但有效的生成视频检测模型，可以消除空间伪影的影响，并表现出强大的泛化能力。

    

    高级视频生成方法产生的视频质量不断提高，这导致社会面临新的安全挑战，使生成视频检测成为紧迫的研究重点。为促进这一领域的合作研究，我们构建了第一个明确用于生成视频检测的开源数据集，为社区提供了一个宝贵的资源，以评估和改进检测方法。通过一系列精心设计的探测实验，我们的研究探讨了时间和空间伪影在开发生成视频的通用和稳健检测器方面的重要性。基于视频帧一致性原则，我们引入了一个简单但有效的检测模型（DeCoF），它消除了空间伪影在通用特征学习中的影响。我们的广泛实验表明，DeCoF在检测未见过的视频生成模型产生的视频方面非常有效，并且验证了其在多个领域的强大泛化能力。

    The escalating quality of video generated by advanced video generation methods leads to new security challenges in society, which makes generated video detection an urgent research priority.To foster collaborative research in this area, we construct the first open-source dataset explicitly for generated video detection, providing a valuable resource for the community to benchmark and improve detection methodologies. Through a series of carefully designed probe experiments, our study explores the significance of temporal and spatial artifacts in developing general and robust detectors for generated video. Based on the principle of video frame consistency, we introduce a simple yet effective detection model (DeCoF) that eliminates the impact of spatial artifacts during generalizing feature learning. Our extensive experiments demonstrate the efficacy of DeCoF in detecting videos produced by unseen video generation models and confirm its powerful generalization capabilities across several 
    
[^6]: 深度学习在多标签学习中的应用：一项全面调研

    Deep Learning for Multi-Label Learning: A Comprehensive Survey. (arXiv:2401.16549v1 [cs.LG])

    [http://arxiv.org/abs/2401.16549](http://arxiv.org/abs/2401.16549)

    深度学习在多标签学习中的综合调研，旨在审视深度学习在解决多标签分类中的挑战方面的最新进展。

    

    多标签学习是一个快速发展的研究领域，旨在从单个输入数据点中预测多个标签。在大数据时代，涉及多标签分类或排名的任务带来了重大而复杂的挑战，在各个领域引起了极大关注。多标签分类面临的困难包括处理高维数据、解决标签相关性和处理部分标签，传统方法在这方面表现不佳。近年来，人们越来越多地采用深度学习技术来更有效地应对多标签分类中的这些挑战。值得注意的是，针对深度学习在多标签学习中的综合研究还比较有限。因此，本调研旨在全面审视深度学习在多标签学习中的最新进展。

    Multi-label learning is a rapidly growing research area that aims to predict multiple labels from a single input data point. In the era of big data, tasks involving multi-label classification (MLC) or ranking present significant and intricate challenges, capturing considerable attention in diverse domains. Inherent difficulties in MLC include dealing with high-dimensional data, addressing label correlations, and handling partial labels, for which conventional methods prove ineffective. Recent years have witnessed a notable increase in adopting deep learning (DL) techniques to address these challenges more effectively in MLC. Notably, there is a burgeoning effort to harness the robust learning capabilities of DL for improved modelling of label dependencies and other challenges in MLC. However, it is noteworthy that comprehensive studies specifically dedicated to DL for multi-label learning are limited. Thus, this survey aims to thoroughly review recent progress in DL for multi-label lea
    
[^7]: ODIN: 一个用于2D和3D感知的单一模型

    ODIN: A Single Model for 2D and 3D Perception. (arXiv:2401.02416v1 [cs.CV])

    [http://arxiv.org/abs/2401.02416](http://arxiv.org/abs/2401.02416)

    ODIN是一个模型，可以同时对2D RGB图像和3D点云进行分割和标记，使用变压器架构进行2D和3D视图间的信息融合。

    

    目前的先进模型在像ScanNet这样的当代3D感知基准上使用并标记依赖于数据集提供的3D点云，该点云是通过对感知到的多视角RGB-D图像进行后处理获得的。它们通常在领域内进行训练，放弃了大规模的2D预训练，并且胜过将姿态RGB-D多视角图像进行特征化的替代方案。消耗姿态图像和后处理的3D点云之间的性能差距，加剧了2D和3D感知需要不同模型架构的观点。在本文中，我们挑战这个观点，并提出ODIN（Omni-Dimensional INstance segmentation），一种能够使用变压器架构对2D RGB图像和3D点云进行分割和标记的模型，该模型通过交替的2D视图内和3D视图间信息融合来区分2D和3D特征操作，利用涉及的令牌的位置编码来捕捉2D补丁令牌和3D坐标的像素坐标。

    State-of-the-art models on contemporary 3D perception benchmarks like ScanNet consume and label dataset-provided 3D point clouds, obtained through post processing of sensed multiview RGB-D images. They are typically trained in-domain, forego large-scale 2D pre-training and outperform alternatives that featurize the posed RGB-D multiview images instead. The gap in performance between methods that consume posed images versus post-processed 3D point clouds has fueled the belief that 2D and 3D perception require distinct model architectures. In this paper, we challenge this view and propose ODIN (Omni-Dimensional INstance segmentation), a model that can segment and label both 2D RGB images and 3D point clouds, using a transformer architecture that alternates between 2D within-view and 3D cross-view information fusion. Our model differentiates 2D and 3D feature operations through the positional encodings of the tokens involved, which capture pixel coordinates for 2D patch tokens and 3D coor
    
[^8]: 利用配对的OpenStreetMap数据和光学高分辨率影像进行土地覆盖变化检测

    Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer. (arXiv:2310.02674v1 [cs.CV])

    [http://arxiv.org/abs/2310.02674](http://arxiv.org/abs/2310.02674)

    本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，提出了一种基于对象引导的Transformer架构，从而拓宽了变化检测任务的范围，并显著减少了计算开销和内存负担。

    

    光学高分辨率影像和OpenStreetMap（OSM）数据是土地覆盖变化检测的两个重要数据源。先前的研究主要利用OSM数据来辅助多时期光学高分辨率图像的变化检测。本文通过直接利用配对的OSM数据和光学图像进行土地覆盖变化检测，拓宽了变化检测任务的范围，涵盖更多动态地球观测。为此，我们提出了一种基于对象引导的Transformer（ObjFormer）架构，将流行的基于对象的图像分析（OBIA）技术与先进的视觉Transformer架构自然地结合起来。引入OBIA可以显著减少自注意力模块中的计算开销和内存负担。具体而言，所提出的ObjFormer具有层次伪孪生编码器，包含对象引导自注意力模块，用于提取代表性特征。

    Optical high-resolution imagery and OpenStreetMap (OSM) data are two important data sources for land-cover change detection. Previous studies in these two data sources focus on utilizing the information in OSM data to aid the change detection on multi-temporal optical high-resolution images. This paper pioneers the direct detection of land-cover changes utilizing paired OSM data and optical imagery, thereby broadening the horizons of change detection tasks to encompass more dynamic earth observations. To this end, we propose an object-guided Transformer (ObjFormer) architecture by naturally combining the prevalent object-based image analysis (OBIA) technique with the advanced vision Transformer architecture. The introduction of OBIA can significantly reduce the computational overhead and memory burden in the self-attention module. Specifically, the proposed ObjFormer has a hierarchical pseudo-siamese encoder consisting of object-guided self-attention modules that extract representative
    
[^9]: 通过最终层反演进行单模型归因

    Single-Model Attribution via Final-Layer Inversion. (arXiv:2306.06210v1 [cs.CV])

    [http://arxiv.org/abs/2306.06210](http://arxiv.org/abs/2306.06210)

    本文提出了一种利用最终层反演和异常检测的开放式单模型归因方法，解决了以往方法要么局限于封闭式环境、要么需要对生成模型进行不必要的改变的问题。实验结果表明该方法优于现有方法。

    

    最近关于生成模型方面的开创性发展引起了人们对于实用单模型归因的兴趣。这些方法可以预测一个样本是由特定的生成器生成的还是不是，例如，为了证明知识产权盗窃行为。然而，以前的方法要么局限于封闭式环境，要么需要对生成模型进行不必要的改变。本文提出了FLIPAD，一种基于最终层反演和异常检测的开放式单模型归因方法，以解决这些问题。我们展示利用的最终层反演可以简化为一个凸的 Lasso 优化问题，从而使我们的方法在理论上可靠且计算效率高。理论结果还得到了实验研究的支持，证明本文方法的有效性，优于现有方法。

    Recent groundbreaking developments on generative modeling have sparked interest in practical single-model attribution. Such methods predict whether a sample was generated by a specific generator or not, for instance, to prove intellectual property theft. However, previous works are either limited to the closed-world setting or require undesirable changes of the generative model. We address these shortcomings by proposing FLIPAD, a new approach for single-model attribution in the open-world setting based on final-layer inversion and anomaly detection. We show that the utilized final-layer inversion can be reduced to a convex lasso optimization problem, making our approach theoretically sound and computationally efficient. The theoretical findings are accompanied by an experimental study demonstrating the effectiveness of our approach, outperforming the existing methods.
    
[^10]: 具有谐振器网络的神经形态视觉场景理解

    Neuromorphic Visual Scene Understanding with Resonator Networks. (arXiv:2208.12880v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2208.12880](http://arxiv.org/abs/2208.12880)

    本论文提出了一种基于神经形态的解决方案，利用高效的因式分解网络来理解视觉场景并推断物体和姿势。关键创新包括基于复值向量的计算框架VSA、用于处理平移和旋转的分层谐振器网络HRN设计，以及在神经形态硬件上实现复值谐振器网络的多组分脉冲相位神经元模型。

    

    理解视觉场景并推断其各个物体的身份和姿势仍然是一个未解决的问题。在这里，我们提出了一种神经形态的解决方案，它利用了基于三个关键概念的高效的因式分解网络：（1）基于复值向量的矢量符号体系架构(VSA)的计算框架；（2）用于处理视觉场景中平移和旋转的非可交换性的分层谐振器网络（HRN）的设计，当两者结合使用时；（3）设计了一种多组分脉冲相位神经元模型，用于在神经形态硬件上实现复值谐振器网络。VSA框架使用矢量绑定操作来产生生成式图像模型，其中绑定作为几何变换的等变操作。因此，一个场景可以被描述为向量乘积的和，而这些向量乘积可以通过谐振器网络的因式分解来高效地推断物体和它们的姿势。

    Understanding a visual scene by inferring identities and poses of its individual objects is still and open problem. Here we propose a neuromorphic solution that utilizes an efficient factorization network based on three key concepts: (1) a computational framework based on Vector Symbolic Architectures (VSA) with complex-valued vectors; (2) the design of Hierarchical Resonator Networks (HRN) to deal with the non-commutative nature of translation and rotation in visual scenes, when both are used in combination; (3) the design of a multi-compartment spiking phasor neuron model for implementing complex-valued resonator networks on neuromorphic hardware. The VSA framework uses vector binding operations to produce generative image models in which binding acts as the equivariant operation for geometric transformations. A scene can therefore be described as a sum of vector products, which in turn can be efficiently factorized by a resonator network to infer objects and their poses. The HRN ena
    

