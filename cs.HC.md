# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^2] | [Large Language Model-Based Interpretable Machine Learning Control in Building Energy Systems](https://arxiv.org/abs/2402.09584) | 本文研究了机器学习控制在建筑能源系统中的可解释性，通过将Shapley值和大型语言模型相结合，提高了机器学习控制模型的透明性和理解性。 |
| [^3] | [Advancing Building Energy Modeling with Large Language Models: Exploration and Case Studies](https://arxiv.org/abs/2402.09579) | 本文研究了将大型语言模型ChatGPT与EnergyPlus建筑能源建模软件融合的创新方法，并强调了大型语言模型在解决建筑能源建模挑战方面的潜力和多种应用。 |

# 详细

[^1]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^2]: 基于大型语言模型的建筑能源系统机器学习控制的可解释性研究

    Large Language Model-Based Interpretable Machine Learning Control in Building Energy Systems

    [https://arxiv.org/abs/2402.09584](https://arxiv.org/abs/2402.09584)

    本文研究了机器学习控制在建筑能源系统中的可解释性，通过将Shapley值和大型语言模型相结合，提高了机器学习控制模型的透明性和理解性。

    

    机器学习控制在暖通空调系统中的潜力受限于其不透明的性质和推理机制，这对于用户和建模者来说是具有挑战性的，难以完全理解，最终导致对基于机器学习控制的决策缺乏信任。为了解决这个挑战，本文研究和探索了可解释机器学习（IML），它是机器学习的一个分支，可以增强模型和推理的透明性和理解性，以提高MLC及其在暖通空调系统中的工业应用的可信度。具体而言，我们开发了一个创新性的框架，将Shapley值的原则和大型语言模型（LLMs）的上下文学习特性相结合。而Shapley值在解剖ML模型中各种特征的贡献方面起到了重要作用，LLM则可以深入理解MLC中基于规则的部分；将它们结合起来，LLM进一步将这些洞见打包到一个

    arXiv:2402.09584v1 Announce Type: new  Abstract: The potential of Machine Learning Control (MLC) in HVAC systems is hindered by its opaque nature and inference mechanisms, which is challenging for users and modelers to fully comprehend, ultimately leading to a lack of trust in MLC-based decision-making. To address this challenge, this paper investigates and explores Interpretable Machine Learning (IML), a branch of Machine Learning (ML) that enhances transparency and understanding of models and their inferences, to improve the credibility of MLC and its industrial application in HVAC systems. Specifically, we developed an innovative framework that combines the principles of Shapley values and the in-context learning feature of Large Language Models (LLMs). While the Shapley values are instrumental in dissecting the contributions of various features in ML models, LLM provides an in-depth understanding of rule-based parts in MLC; combining them, LLM further packages these insights into a
    
[^3]: 用大型语言模型推动建筑能源建模：探索和案例研究

    Advancing Building Energy Modeling with Large Language Models: Exploration and Case Studies

    [https://arxiv.org/abs/2402.09579](https://arxiv.org/abs/2402.09579)

    本文研究了将大型语言模型ChatGPT与EnergyPlus建筑能源建模软件融合的创新方法，并强调了大型语言模型在解决建筑能源建模挑战方面的潜力和多种应用。

    

    人工智能的快速发展促进了像ChatGPT这样的大型语言模型的出现，为专门的工程建模（尤其是基于物理的建筑能源建模）提供了潜在的应用。本文研究了将大型语言模型与建筑能源建模软件（具体为EnergyPlus）融合的创新方法。首先进行了文献综述，揭示了在工程建模中整合大型语言模型的增长趋势，但在建筑能源建模中的应用研究仍然有限。我们强调了大型语言模型在解决建筑能源建模挑战方面的潜力，并概述了潜在的应用，包括：1）模拟输入生成，2）模拟输出分析和可视化，3）进行错误分析，4）共模拟，5）模拟知识提取。

    arXiv:2402.09579v1 Announce Type: cross  Abstract: The rapid progression in artificial intelligence has facilitated the emergence of large language models like ChatGPT, offering potential applications extending into specialized engineering modeling, especially physics-based building energy modeling. This paper investigates the innovative integration of large language models with building energy modeling software, focusing specifically on the fusion of ChatGPT with EnergyPlus. A literature review is first conducted to reveal a growing trend of incorporating of large language models in engineering modeling, albeit limited research on their application in building energy modeling. We underscore the potential of large language models in addressing building energy modeling challenges and outline potential applications including 1) simulation input generation, 2) simulation output analysis and visualization, 3) conducting error analysis, 4) co-simulation, 5) simulation knowledge extraction a
    

