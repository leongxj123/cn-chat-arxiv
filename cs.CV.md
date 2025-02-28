# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^2] | [Advancing Low-Rank and Local Low-Rank Matrix Approximation in Medical Imaging: A Systematic Literature Review and Future Directions](https://arxiv.org/abs/2402.14045) | 本文系统综述了在医学成像中应用低秩矩阵逼近（LRMA）和其派生物局部LRMA（LLRMA）的作品，并指出自2015年以来医学成像领域开始偏向于使用LLRMA，显示其在捕获医学数据中复杂结构方面的潜力和有效性。 |
| [^3] | [Elucidating the solution space of extended reverse-time SDE for diffusion models.](http://arxiv.org/abs/2309.06169) | 这项工作介绍了扩展反向时间随机微分方程（ER SDE）用于解决扩散模型中的采样问题，并提供了精确解和高阶近似解，并解释了在快速采样方面ODE求解器优于SDE求解器的数学洞察力。 |

# 详细

[^1]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^2]: 在医学成像中推进低秩和局部低秩矩阵逼近：系统文献综述与未来方向

    Advancing Low-Rank and Local Low-Rank Matrix Approximation in Medical Imaging: A Systematic Literature Review and Future Directions

    [https://arxiv.org/abs/2402.14045](https://arxiv.org/abs/2402.14045)

    本文系统综述了在医学成像中应用低秩矩阵逼近（LRMA）和其派生物局部LRMA（LLRMA）的作品，并指出自2015年以来医学成像领域开始偏向于使用LLRMA，显示其在捕获医学数据中复杂结构方面的潜力和有效性。

    

    医学成像数据集的大容量和复杂性是存储、传输和处理的瓶颈。为解决这些挑战，低秩矩阵逼近（LRMA）及其派生物局部LRMA（LLRMA）的应用已显示出潜力。本文进行了系统文献综述，展示了在医学成像中应用LRMA和LLRMA的作品。文献的详细分析确认了应用于各种成像模态的LRMA和LLRMA方法。本文解决了现有LRMA和LLRMA方法所面临的挑战和限制。我们注意到，自2015年以来，医学成像领域明显偏向于LLRMA，显示了相对于LRMA在捕获医学数据中复杂结构方面的潜力和有效性。鉴于LLRMA所使用的浅层相似性方法的限制，我们建议使用先进语义图像分割来处理相似性。

    arXiv:2402.14045v1 Announce Type: cross  Abstract: The large volume and complexity of medical imaging datasets are bottlenecks for storage, transmission, and processing. To tackle these challenges, the application of low-rank matrix approximation (LRMA) and its derivative, local LRMA (LLRMA) has demonstrated potential.   This paper conducts a systematic literature review to showcase works applying LRMA and LLRMA in medical imaging. A detailed analysis of the literature identifies LRMA and LLRMA methods applied to various imaging modalities. This paper addresses the challenges and limitations associated with existing LRMA and LLRMA methods.   We note a significant shift towards a preference for LLRMA in the medical imaging field since 2015, demonstrating its potential and effectiveness in capturing complex structures in medical data compared to LRMA. Acknowledging the limitations of shallow similarity methods used with LLRMA, we suggest advanced semantic image segmentation for similarit
    
[^3]: 阐明扩展反向时间随机微分方程在扩散模型中的解空间

    Elucidating the solution space of extended reverse-time SDE for diffusion models. (arXiv:2309.06169v1 [cs.LG])

    [http://arxiv.org/abs/2309.06169](http://arxiv.org/abs/2309.06169)

    这项工作介绍了扩展反向时间随机微分方程（ER SDE）用于解决扩散模型中的采样问题，并提供了精确解和高阶近似解，并解释了在快速采样方面ODE求解器优于SDE求解器的数学洞察力。

    

    扩散模型在各种生成建模任务中展示出强大的图像生成能力。然而，它们的主要限制在于采样速度较慢，需要通过大型神经网络进行数百或数千次连续函数评估才能生成高质量的图像。从扩散模型中采样可以看作是解相应的随机微分方程（SDE）或常微分方程（ODE）。在这项工作中，我们将采样过程形式化为扩展反向时间 SDE（ER SDE），将之前对ODE和SDE的探索统一起来。利用ER SDE解的半线性结构，我们为VP SDE提供了精确解和任意高阶近似解，为VE SDE提供了高阶近似解。基于ER SDE的解空间，我们揭示了ODE求解器在快速采样方面优于SDE求解器的数学洞察力。此外，我们还揭示了VP SDE求解器与其VE SDE求解器在性能上相当。

    Diffusion models (DMs) demonstrate potent image generation capabilities in various generative modeling tasks. Nevertheless, their primary limitation lies in slow sampling speed, requiring hundreds or thousands of sequential function evaluations through large neural networks to generate high-quality images. Sampling from DMs can be seen as solving corresponding stochastic differential equations (SDEs) or ordinary differential equations (ODEs). In this work, we formulate the sampling process as an extended reverse-time SDE (ER SDE), unifying prior explorations into ODEs and SDEs. Leveraging the semi-linear structure of ER SDE solutions, we offer exact solutions and arbitrarily high-order approximate solutions for VP SDE and VE SDE, respectively. Based on the solution space of the ER SDE, we yield mathematical insights elucidating the superior performance of ODE solvers over SDE solvers in terms of fast sampling. Additionally, we unveil that VP SDE solvers stand on par with their VE SDE c
    

