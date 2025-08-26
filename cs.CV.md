# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout](https://arxiv.org/abs/2404.00412) | SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。 |
| [^2] | [Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach.](http://arxiv.org/abs/2401.09671) | 本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。 |

# 详细

[^1]: SVGCraft:超越单个目标文字到SVG综合画布布局

    SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout

    [https://arxiv.org/abs/2404.00412](https://arxiv.org/abs/2404.00412)

    SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。

    

    生成从文本提示到矢量图的VectorArt是一项具有挑战性的视觉任务，需要对已知和未知实体进行多样化而真实的描述。然而，现有研究主要局限于生成单个对象，而不是由多个元素组成的场景。为此，本文介绍了SVGCraft，这是一个新颖的端到端框架，用于从文本描述中生成描绘整个场景的矢量图。该框架利用预训练的LLM从文本提示生成布局，并引入了一种技术，通过生产特定边界框中的掩膜潜变量实现准确的对象放置。它引入了一个融合机制，用于集成注意力图，并使用扩散U-Net进行连贯的合成，加快绘图过程。生成的SVG使用预训练的编码器和LPIPS损失进行优化，通过透明度调制来最大程度地增加相似性。

    arXiv:2404.00412v1 Announce Type: cross  Abstract: Generating VectorArt from text prompts is a challenging vision task, requiring diverse yet realistic depictions of the seen as well as unseen entities. However, existing research has been mostly limited to the generation of single objects, rather than comprehensive scenes comprising multiple elements. In response, this work introduces SVGCraft, a novel end-to-end framework for the creation of vector graphics depicting entire scenes from textual descriptions. Utilizing a pre-trained LLM for layout generation from text prompts, this framework introduces a technique for producing masked latents in specified bounding boxes for accurate object placement. It introduces a fusion mechanism for integrating attention maps and employs a diffusion U-Net for coherent composition, speeding up the drawing process. The resulting SVG is optimized using a pre-trained encoder and LPIPS loss with opacity modulation to maximize similarity. Additionally, th
    
[^2]: 迈向可识别的无监督领域转换：一种多样化分布匹配的方法

    Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach. (arXiv:2401.09671v1 [cs.LG])

    [http://arxiv.org/abs/2401.09671](http://arxiv.org/abs/2401.09671)

    本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。

    

    无监督领域转换（UDT）旨在找到将一个领域的样本（例如素描）转换为另一个领域（例如照片）的函数，同时不改变高层语义意义（也称为“内容”）。这些转换函数通常通过转换源领域和目标领域的概率分布来寻找。CycleGAN可以说是这一领域中最具代表性的方法。然而，文献中指出CycleGAN及其变体可能无法识别所需的转换函数，并产生内容不对齐的转换。这种局限性源于学习准则解空间中存在多个转换函数，称为“保度自同构（MPA）”。尽管意识到了这种可识别性问题，但解决方案仍然难以找到。本研究深入探究了核心的可识别性问题，并引入了MPA消除理论。我们的分析表明...

    Unsupervised domain translation (UDT) aims to find functions that convert samples from one domain (e.g., sketches) to another domain (e.g., photos) without changing the high-level semantic meaning (also referred to as ``content''). The translation functions are often sought by probability distribution matching of the transformed source domain and target domain. CycleGAN stands as arguably the most representative approach among this line of work. However, it was noticed in the literature that CycleGAN and variants could fail to identify the desired translation functions and produce content-misaligned translations. This limitation arises due to the presence of multiple translation functions -- referred to as ``measure-preserving automorphism" (MPA) -- in the solution space of the learning criteria. Despite awareness of such identifiability issues, solutions have remained elusive. This study delves into the core identifiability inquiry and introduces an MPA elimination theory. Our analysi
    

