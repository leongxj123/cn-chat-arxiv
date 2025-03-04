# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Reinforcement Learning for Controlled Traversing of the Attractor Landscape of Boolean Models in the Context of Cellular Reprogramming](https://arxiv.org/abs/2402.08491) | 本研究开发了一个基于深度强化学习的计算框架，用于细胞重编程中的重编程策略识别。在控制问题中，引入了伪吸引子的概念和识别方法，并设计了一个用于解决该问题的计算框架。 |
| [^2] | [Open-radiomics: A Collection of Standardized Datasets and a Technical Protocol for Reproducible Radiomics Machine Learning Pipelines.](http://arxiv.org/abs/2207.14776) | 本研究提出了一套开放放射组学数据集和技术协议，旨在解决放射组学在结果可重复性和可访问性方面所面临的挑战。通过在BraTS 2020数据集上进行实验，研究了放射组学特征提取对结果可重复性的影响。 |

# 详细

[^1]: 深度强化学习在细胞重编程的布尔模型吸引子景观中的控制遍历中的应用研究

    Deep Reinforcement Learning for Controlled Traversing of the Attractor Landscape of Boolean Models in the Context of Cellular Reprogramming

    [https://arxiv.org/abs/2402.08491](https://arxiv.org/abs/2402.08491)

    本研究开发了一个基于深度强化学习的计算框架，用于细胞重编程中的重编程策略识别。在控制问题中，引入了伪吸引子的概念和识别方法，并设计了一个用于解决该问题的计算框架。

    

    细胞重编程可用于预防和治疗不同疾病。然而，通过传统湿实验发现重编程策略的效率受到时间和成本的限制。在本研究中，我们基于深度强化学习开发了一个新颖的计算框架，以便帮助识别重编程策略。为此，我们在细胞重编程框架的BNs和PBNs以及异步更新模式下制定了一个控制问题。此外，我们引入了伪吸引子的概念和训练过程中伪吸引子状态的识别方法。最后，我们设计了一个用于解决控制问题的计算框架，并在多个不同模型上进行了测试。

    Cellular reprogramming can be used for both the prevention and cure of different diseases. However, the efficiency of discovering reprogramming strategies with classical wet-lab experiments is hindered by lengthy time commitments and high costs. In this study, we develop a~novel computational framework based on deep reinforcement learning that facilitates the identification of reprogramming strategies. For this aim, we formulate a~control problem in the context of cellular reprogramming for the frameworks of BNs and PBNs under the asynchronous update mode. Furthermore, we introduce the notion of a~pseudo-attractor and a~procedure for identification of pseudo-attractor state during training. Finally, we devise a~computational framework for solving the control problem, which we test on a~number of different models.
    
[^2]: 开放放射组学：一系列标准化数据集和可重复放射组学机器学习流程的技术协议

    Open-radiomics: A Collection of Standardized Datasets and a Technical Protocol for Reproducible Radiomics Machine Learning Pipelines. (arXiv:2207.14776v2 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2207.14776](http://arxiv.org/abs/2207.14776)

    本研究提出了一套开放放射组学数据集和技术协议，旨在解决放射组学在结果可重复性和可访问性方面所面临的挑战。通过在BraTS 2020数据集上进行实验，研究了放射组学特征提取对结果可重复性的影响。

    

    目的：作为医学影像中机器学习流程的一个重要分支，放射组学面临着两个主要挑战，即可重复性和可访问性。在这项工作中，我们介绍了开放放射组学，一套放射组学数据集以及基于我们提出的技术协议的综合放射组学流程，以研究放射组学特征提取对结果可重复性的影响。材料和方法：实验使用BraTS 2020开源磁共振成像（MRI）数据集进行，包括369名患有脑肿瘤的成年患者（76例低级别胶质瘤（LGG）和293例高级别胶质瘤（HGG））。使用PyRadiomics库进行LGG与HGG分类，形成了288个放射组学数据集；其中包括4个MRI序列、3个binWidths、6种图像归一化方法和4个肿瘤次区域的组合。使用随机森林分类器，并为每个放射组学数据集进行训练-验证-测试（60%/20%/20%）实验，采用不同的数据划分和m

    Purpose: As an important branch of machine learning pipelines in medical imaging, radiomics faces two major challenges namely reproducibility and accessibility. In this work, we introduce open-radiomics, a set of radiomics datasets along with a comprehensive radiomics pipeline based on our proposed technical protocol to investigate the effects of radiomics feature extraction on the reproducibility of the results.  Materials and Methods: Experiments are conducted on BraTS 2020 open-source Magnetic Resonance Imaging (MRI) dataset that includes 369 adult patients with brain tumors (76 low-grade glioma (LGG), and 293 high-grade glioma (HGG)). Using PyRadiomics library for LGG vs. HGG classification, 288 radiomics datasets are formed; the combinations of 4 MRI sequences, 3 binWidths, 6 image normalization methods, and 4 tumor subregions.  Random Forest classifiers were used, and for each radiomics dataset the training-validation-test (60%/20%/20%) experiment with different data splits and m
    

