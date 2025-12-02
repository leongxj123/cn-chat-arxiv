# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout](https://arxiv.org/abs/2404.00412) | SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。 |
| [^2] | [GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation](https://arxiv.org/abs/2403.07247) | 该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。 |

# 详细

[^1]: SVGCraft:超越单个目标文字到SVG综合画布布局

    SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout

    [https://arxiv.org/abs/2404.00412](https://arxiv.org/abs/2404.00412)

    SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。

    

    生成从文本提示到矢量图的VectorArt是一项具有挑战性的视觉任务，需要对已知和未知实体进行多样化而真实的描述。然而，现有研究主要局限于生成单个对象，而不是由多个元素组成的场景。为此，本文介绍了SVGCraft，这是一个新颖的端到端框架，用于从文本描述中生成描绘整个场景的矢量图。该框架利用预训练的LLM从文本提示生成布局，并引入了一种技术，通过生产特定边界框中的掩膜潜变量实现准确的对象放置。它引入了一个融合机制，用于集成注意力图，并使用扩散U-Net进行连贯的合成，加快绘图过程。生成的SVG使用预训练的编码器和LPIPS损失进行优化，通过透明度调制来最大程度地增加相似性。

    arXiv:2404.00412v1 Announce Type: cross  Abstract: Generating VectorArt from text prompts is a challenging vision task, requiring diverse yet realistic depictions of the seen as well as unseen entities. However, existing research has been mostly limited to the generation of single objects, rather than comprehensive scenes comprising multiple elements. In response, this work introduces SVGCraft, a novel end-to-end framework for the creation of vector graphics depicting entire scenes from textual descriptions. Utilizing a pre-trained LLM for layout generation from text prompts, this framework introduces a technique for producing masked latents in specified bounding boxes for accurate object placement. It introduces a fusion mechanism for integrating attention maps and employs a diffusion U-Net for coherent composition, speeding up the drawing process. The resulting SVG is optimized using a pre-trained encoder and LPIPS loss with opacity modulation to maximize similarity. Additionally, th
    
[^2]: GuideGen：一种用于联合CT体积和解剖结构生成的文本引导框架

    GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation

    [https://arxiv.org/abs/2403.07247](https://arxiv.org/abs/2403.07247)

    该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。

    

    arXiv:2403.07247v1 公告类型：交叉 摘要：为了收集带有图像和相应标签的大型医学数据集而进行的注释负担和大量工作很少是划算且令人望而生畏的。这导致了缺乏丰富的训练数据，削弱了下游任务，并在一定程度上加剧了医学领域面临的图像分析挑战。作为一种权宜之计，鉴于生成性神经模型的最近成功，现在可以在外部约束的引导下以高保真度合成图像数据集。本文探讨了这种可能性，并提出了GuideGen：一种联合生成腹部器官和结直肠癌CT图像和组织掩膜的管线，其受文本提示条件约束。首先，我们介绍了体积掩膜采样器，以适应掩膜标签的离散分布并生成低分辨率3D组织掩膜。其次，我们的条件图像生成器会在收到相应文本提示的情况下自回归生成CT切片。

    arXiv:2403.07247v1 Announce Type: cross  Abstract: The annotation burden and extensive labor for gathering a large medical dataset with images and corresponding labels are rarely cost-effective and highly intimidating. This results in a lack of abundant training data that undermines downstream tasks and partially contributes to the challenge image analysis faces in the medical field. As a workaround, given the recent success of generative neural models, it is now possible to synthesize image datasets at a high fidelity guided by external constraints. This paper explores this possibility and presents \textbf{GuideGen}: a pipeline that jointly generates CT images and tissue masks for abdominal organs and colorectal cancer conditioned on a text prompt. Firstly, we introduce Volumetric Mask Sampler to fit the discrete distribution of mask labels and generate low-resolution 3D tissue masks. Secondly, our Conditional Image Generator autoregressively generates CT slices conditioned on a corre
    

