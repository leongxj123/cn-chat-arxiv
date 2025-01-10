# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.](http://arxiv.org/abs/2303.00915) | BiomedCLIP是一个从1500万科学图像-文本对中预训练的多模态生物医学基础模型，其基于大规模的PMC-15M数据集进行训练，该数据集比现有的生物医学多模态数据集大两个数量级，并成功应用于生物医学图像任务的检索、分类和视觉问题回答等方面。 |

# 详细

[^1]: BiomedCLIP：一种从一千五百万科学图像-文本对进行预训练的多模态生物医学基础模型

    BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. (arXiv:2303.00915v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.00915](http://arxiv.org/abs/2303.00915)

    BiomedCLIP是一个从1500万科学图像-文本对中预训练的多模态生物医学基础模型，其基于大规模的PMC-15M数据集进行训练，该数据集比现有的生物医学多模态数据集大两个数量级，并成功应用于生物医学图像任务的检索、分类和视觉问题回答等方面。

    

    生物医学数据本质上是多模态的，包括物理测量和自然语言叙述。一个通用的生物医学人工智能模型需要同时处理不同的数据模态，包括文本和图像。因此，训练一个有效的通用生物医学模型需要高质量的多模态数据，例如平行的图像-文本对。在这里，我们提供了一个新颖的数据集PMC-15M，比现有的生物医学多模态数据集（如MIMIC-CXR）大两个数量级，并涵盖了各种各样的生物医学图像类型。PMC-15M包含了来自440万科学论文的1500万个生物医学图像-文本对。基于PMC-15M，我们训练了BiomedCLIP，一个多模态基础模型，并进行了领域特定的自适应，以适用于生物医学视觉-语言处理。我们在标准的生物医学图像任务，从检索到分类到视觉问题回答（VQA）方面进行了大量的实验和消融研究。

    Biomedical data is inherently multimodal, comprising physical measurements and natural language narratives. A generalist biomedical AI model needs to simultaneously process different modalities of data, including text and images. Therefore, training an effective generalist biomedical model requires high-quality multimodal data, such as parallel image-text pairs. Here, we present PMC-15M, a novel dataset that is two orders of magnitude larger than existing biomedical multimodal datasets such as MIMIC-CXR, and spans a diverse range of biomedical image types. PMC-15M contains 15 million biomedical image-text pairs collected from 4.4 million scientific articles. Based on PMC-15M, we have pretrained BiomedCLIP, a multimodal foundation model, with domain-specific adaptations tailored to biomedical vision-language processing. We conducted extensive experiments and ablation studies on standard biomedical imaging tasks from retrieval to classification to visual question-answering (VQA). BiomedC
    

