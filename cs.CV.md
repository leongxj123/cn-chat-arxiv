# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^2] | [GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure](https://arxiv.org/abs/2311.11319) | GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。 |
| [^3] | [A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays.](http://arxiv.org/abs/2310.17176) | 本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。 |

# 详细

[^1]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^2]: GeoSAM: 使用稀疏和密集的视觉提示对SAM进行改进，实现自动化的移动基础设施分割

    GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure

    [https://arxiv.org/abs/2311.11319](https://arxiv.org/abs/2311.11319)

    GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。

    

    当应用于自然图像分割时，Segment Anything Model (SAM)已经展现出了令人印象深刻的性能。然而，它在地理图像（如航拍和卫星图像）中面临困难，特别是在分割道路、人行道和人行横道等移动基础设施时。这种较差的性能源于这些对象的窄小特征，它们的纹理融入环境中，以及树木、建筑物、车辆和行人等物体的干扰，这些都可能使模型失去定向产生不准确的分割图。为了解决这些挑战，我们提出了地理SAM（GeoSAM），这是一个基于SAM的新框架，它使用来自零样本学习的密集视觉提示和预训练CNN分割模型的稀疏视觉提示实施了细调策略。所提出的GeoSAM在地理图像分割方面优于现有方法，特别是对于道路基础设施、行人基础设施的分割性能提升了26％、7％和17％。

    The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects, their textures blending into the surroundings, and interference from objects like trees, buildings, vehicles, and pedestrians - all of which can disorient the model to produce inaccurate segmentation maps. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning strategy using the dense visual prompt from zero-shot learning, and the sparse visual prompt from a pre-trained CNN segmentation model. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 26%, 7%, and 17% for road infrastructure, pedestrian infrastructur
    
[^3]: 从全景X射线中进行牙齿分割和定位的深度学习方法

    A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays. (arXiv:2310.17176v1 [cs.CV])

    [http://arxiv.org/abs/2310.17176](http://arxiv.org/abs/2310.17176)

    本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。

    

    准确的牙齿分割和定位在现代口腔保健中是基础，可实现精确诊断、治疗计划和牙齿种植设计。本研究提出了一种综合的方法，利用深度学习技术从全景X射线图像中进行牙齿分割和定位。我们根据FUSegNet构建了我们的模型，这是一种最初用于创面分割的流行模型，并通过将基于网格的注意力门引入跳跃连接进行了修改。我们通过主成分分析（PCA）引入定向边界框（OBB）生成，以实现精确的牙齿定位估计。在公开可获得的DNS数据集上评估我们的方法，该数据集包括543个全景X射线图像，我们在牙齿实例分割中得到了最高的交并比（IoU）得分82.43%，Dice相似系数（DSC）得分90.37%，在OBB分析中，我们获得了旋转的交并比（RIoU）得分82.82%。

    Accurate teeth segmentation and orientation are fundamental in modern oral healthcare, enabling precise diagnosis, treatment planning, and dental implant design. In this study, we present a comprehensive approach to teeth segmentation and orientation from panoramic X-ray images, leveraging deep learning techniques. We build our model based on FUSegNet, a popular model originally developed for wound segmentation, and introduce modifications by incorporating grid-based attention gates into the skip connections. We introduce oriented bounding box (OBB) generation through principal component analysis (PCA) for precise tooth orientation estimation. Evaluating our approach on the publicly available DNS dataset, comprising 543 panoramic X-ray images, we achieve the highest Intersection-over-Union (IoU) score of 82.43% and Dice Similarity Coefficient (DSC) score of 90.37% among compared models in teeth instance segmentation. In OBB analysis, we obtain the Rotated IoU (RIoU) score of 82.82%. We
    

