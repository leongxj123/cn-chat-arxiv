# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Polyhedral Complex Derivation from Piecewise Trilinear Networks](https://arxiv.org/abs/2402.10403) | 本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。 |
| [^2] | [Mitigating Open-Vocabulary Caption Hallucinations](https://arxiv.org/abs/2312.03631) | 提出了在开放词汇设置中解决图像字幕幻觉问题的框架，并提出了一种新方法MOCHa来缓解幻觉 |
| [^3] | [Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks.](http://arxiv.org/abs/2309.17329) | 本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。 |
| [^4] | [Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception.](http://arxiv.org/abs/2308.05822) | 本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。 |

# 详细

[^1]: 从分段三线性网络中导出多面体复合体

    Polyhedral Complex Derivation from Piecewise Trilinear Networks

    [https://arxiv.org/abs/2402.10403](https://arxiv.org/abs/2402.10403)

    本文以三线性插值方法作为位置编码，提出了理论见解和分析网格提取方法，将高维曲面转换为平面，并引入了一种近似交点的方法，拓展了更广泛的应用。

    

    最近关于深度神经网络可视化的进展揭示了它们结构的见解，并且可以从连续分段仿射（CPWA）函数中提取网格。与此同时，神经表面表示学习的发展包括非线性位置编码，解决了诸如谱偏差之类的问题；然而，这在应用基于CPWA函数的网格提取技术方面带来了挑战。我们聚焦于三线性插值方法作为位置编码，提供了理论见解和分析的网格提取，展示了在奇拿尔约束下将高维曲面转换为三线性区域内的平面的过程。此外，我们引入了一种方法来近似三个高维曲面之间的交点，从而扩展了更广泛的应用。通过汉明距离和效率以及角距离来经验性地验证正确性和简洁性，同时检查了t之间的相关性

    arXiv:2402.10403v1 Announce Type: cross  Abstract: Recent advancements in visualizing deep neural networks provide insights into their structures and mesh extraction from Continuous Piecewise Affine (CPWA) functions. Meanwhile, developments in neural surface representation learning incorporate non-linear positional encoding, addressing issues like spectral bias; however, this poses challenges in applying mesh extraction techniques based on CPWA functions. Focusing on trilinear interpolating methods as positional encoding, we present theoretical insights and an analytical mesh extraction, showing the transformation of hypersurfaces to flat planes within the trilinear region under the eikonal constraint. Moreover, we introduce a method for approximating intersecting points among three hypersurfaces contributing to broader applications. We empirically validate correctness and parsimony through chamfer distance and efficiency, and angular distance, while examining the correlation between t
    
[^2]: 缓解开放词汇描述幻觉

    Mitigating Open-Vocabulary Caption Hallucinations

    [https://arxiv.org/abs/2312.03631](https://arxiv.org/abs/2312.03631)

    提出了在开放词汇设置中解决图像字幕幻觉问题的框架，并提出了一种新方法MOCHa来缓解幻觉

    

    近年来，图像条件的文本生成取得了快速进展，但图像字幕仍然存在幻觉的基本问题，即生成与给定图像无法推断的虚假细节。现有方法在图像字幕中大多使用封闭词汇对象列表来缓解或评估幻觉，忽略了实践中发生的大多数幻觉类型。为此，我们提出了一个框架，以应对开放词汇设置中图像字幕中的幻觉，包括量化它们的存在并优化以减轻这种幻觉。我们的OpenCHAIR基准利用生成基础模型来评估开放词汇描述幻觉，在多样性和准确性方面都超过了流行的CHAIR基准。为了在序列级别上缓解开放词汇的幻觉，我们提出了MOCHa，一种利用进展的方法

    arXiv:2312.03631v2 Announce Type: replace-cross  Abstract: While recent years have seen rapid progress in image-conditioned text generation, image captioning still suffers from the fundamental issue of hallucinations, namely, the generation of spurious details that cannot be inferred from the given image. Existing methods largely use closed-vocabulary object lists to mitigate or evaluate hallucinations in image captioning, ignoring most types of hallucinations that occur in practice. To this end, we propose a framework for addressing hallucinations in image captioning in the open-vocabulary setting, including quantifying their presence and optimizing to mitigate such hallucinations. Our OpenCHAIR benchmark leverages generative foundation models to evaluate open-vocabulary caption hallucinations, surpassing the popular CHAIR benchmark in both diversity and accuracy. To mitigate open-vocabulary hallucinations at the sequence level, we propose MOCHa, an approach harnessing advancements in
    
[^3]: 通过隐式点图网络高效解剖标记肺部树状结构

    Efficient Anatomical labeling of Pulmonary Tree Structures via Implicit Point-Graph Networks. (arXiv:2309.17329v1 [cs.CV])

    [http://arxiv.org/abs/2309.17329](http://arxiv.org/abs/2309.17329)

    本文介绍了一种通过隐式点图网络高效解剖标记肺部树状结构的方法，提供了SOTA准确度和可用的表面，同时还提供了一个用于评估该方法的数据集。

    

    肺部疾病在全球范围内是导致死亡的主要原因之一。治愈肺部疾病需要更好地理解肺部系统内的许多复杂的3D树状结构，如气道、动脉和静脉。在理论上，它们可以通过高分辨率图像堆栈进行建模。然而，基于密集体素网格的标准CNN方法代价过高。为了解决这个问题，我们引入了一种基于点的方法，保留了树骨架的图连通性，并结合了隐式表面表示。它以较低的计算成本提供了SOTA准确度，生成的模型具有可用的表面。由于公开可访问的数据稀缺，我们还整理了一套广泛的数据集来评估我们的方法，并将其公开。

    Pulmonary diseases rank prominently among the principal causes of death worldwide. Curing them will require, among other things, a better understanding of the many complex 3D tree-shaped structures within the pulmonary system, such as airways, arteries, and veins. In theory, they can be modeled using high-resolution image stacks. Unfortunately, standard CNN approaches operating on dense voxel grids are prohibitively expensive. To remedy this, we introduce a point-based approach that preserves graph connectivity of tree skeleton and incorporates an implicit surface representation. It delivers SOTA accuracy at a low computational cost and the resulting models have usable surfaces. Due to the scarcity of publicly accessible data, we have also curated an extensive dataset to evaluate our approach and will make it public.
    
[^4]: 编码-存储-检索：通过语言编码的自我中心感知增强记忆

    Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception. (arXiv:2308.05822v1 [cs.CV])

    [http://arxiv.org/abs/2308.05822](http://arxiv.org/abs/2308.05822)

    本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。

    

    我们依赖于自己的记忆来编码、存储和检索我们的经历。然而，记忆间隔有时会发生。实现记忆增强的一种有希望的方法是通过使用增强现实头戴式显示设备来捕捉和保留自我中心的视频，这种做法通常被称为生活记录。然而，由于当前技术缺乏高效编码和存储如此大量的视频数据的能力，从庞大的视频存档中检索特定信息需要大量的计算能力，进一步复杂了快速访问所需内容的任务。

    We depend on our own memory to encode, store, and retrieve our experiences. However, memory lapses can occur. One promising avenue for achieving memory augmentation is through the use of augmented reality head-mounted displays to capture and preserve egocentric videos, a practice commonly referred to as life logging. However, a significant challenge arises from the sheer volume of video data generated through life logging, as the current technology lacks the capability to encode and store such large amounts of data efficiently. Further, retrieving specific information from extensive video archives requires substantial computational power, further complicating the task of quickly accessing desired content. To address these challenges, we propose a memory augmentation system that involves leveraging natural language encoding for video data and storing them in a vector database. This approach harnesses the power of large vision language models to perform the language encoding process. Add
    

