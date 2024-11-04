# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-Invasive Medical Digital Twins using Physics-Informed Self-Supervised Learning](https://arxiv.org/abs/2403.00177) | 提出了一种使用物理知识的自监督学习算法，通过仅使用非侵入式患者健康数据识别数字孪生体模型参数，从而实现了非侵入式医学数字孪生体的构建。 |

# 详细

[^1]: 使用物理知识的自监督学习构建非侵入式医学数字孪生体

    Non-Invasive Medical Digital Twins using Physics-Informed Self-Supervised Learning

    [https://arxiv.org/abs/2403.00177](https://arxiv.org/abs/2403.00177)

    提出了一种使用物理知识的自监督学习算法，通过仅使用非侵入式患者健康数据识别数字孪生体模型参数，从而实现了非侵入式医学数字孪生体的构建。

    

    数字孪生体是实现实际物理现象的虚拟复制品，利用数学建模来表征和模拟其定义特征。通过为疾病过程构建数字孪生体，我们可以进行仿真，模拟患者在虚拟环境中的健康状况和在假设干预下的对照结果。这消除了侵入性程序或不确定治疗决策的需求。本文提出了一种仅利用非侵入式患者健康数据来识别数字孪生体模型参数的方法。我们将数字孪生体建模看作一个复合逆问题，并观察到其结构类似于自监督学习中的预训练和微调。利用这一点，我们引入了一种基于物理知识的自监督学习算法，这种算法首先在解决物理模型方程的假定任务上对神经网络进行预训练。随后，该模型被训练以...

    arXiv:2403.00177v1 Announce Type: new  Abstract: A digital twin is a virtual replica of a real-world physical phenomena that uses mathematical modeling to characterize and simulate its defining features. By constructing digital twins for disease processes, we can perform in-silico simulations that mimic patients' health conditions and counterfactual outcomes under hypothetical interventions in a virtual setting. This eliminates the need for invasive procedures or uncertain treatment decisions. In this paper, we propose a method to identify digital twin model parameters using only noninvasive patient health data. We approach the digital twin modeling as a composite inverse problem, and observe that its structure resembles pretraining and finetuning in self-supervised learning (SSL). Leveraging this, we introduce a physics-informed SSL algorithm that initially pretrains a neural network on the pretext task of solving the physical model equations. Subsequently, the model is trained to rec
    

