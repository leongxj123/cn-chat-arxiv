# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach](https://arxiv.org/abs/2403.10112) | 本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。 |
| [^2] | [Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management](https://arxiv.org/abs/2402.07949) | 通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。 |
| [^3] | [Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks.](http://arxiv.org/abs/2303.10225) | 本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。 |

# 详细

[^1]: 单智能体与多智能体的私密主动感知：深度神经进化方法

    Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach

    [https://arxiv.org/abs/2403.10112](https://arxiv.org/abs/2403.10112)

    本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。

    

    本文关注存在窥视者情况下的主动假设测试中的一个集中式问题和一个分散式问题。针对包括单个合法智能体的集中式问题，我们提出了基于神经进化（NE）的新框架；而针对分散式问题，我们开发了一种新颖的基于NE的方法，用于解决协作多智能体任务，这种方法有趣地保持了单一智能体NE的所有计算优势。通过对无线传感器网络上异常检测示例用例中的数值实验，验证了所提出的EAHT方法优于传统的主动假设测试策略以及基于学习的方法。

    arXiv:2403.10112v1 Announce Type: new  Abstract: In this paper, we focus on one centralized and one decentralized problem of active hypothesis testing in the presence of an eavesdropper. For the centralized problem including a single legitimate agent, we present a new framework based on NeuroEvolution (NE), whereas, for the decentralized problem, we develop a novel NE-based method for solving collaborative multi-agent tasks, which interestingly maintains all computational benefits of single-agent NE. The superiority of the proposed EAHT approaches over conventional active hypothesis testing policies, as well as learning-based methods, is validated through numerical investigations in an example use case of anomaly detection over wireless sensor networks.
    
[^2]: 优化人工胰腺设计以改善糖尿病管理

    Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management

    [https://arxiv.org/abs/2402.07949](https://arxiv.org/abs/2402.07949)

    通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。

    

    糖尿病是一种慢性疾病，影响美国境内有3800万人，它会影响身体将食物转化为能量（即血糖）的能力。标准的治疗方法是通过使用人工胰腺，即持续胰岛素泵（基础注射），以及定期注射胰岛素（突发注射）来补充碳水化合物摄入量。治疗目标是将血糖保持在可接受范围的中心位置，通过持续血糖测量来进行衡量。次要目标是减少注射次数，因为对某些患者来说注射是不愉快且难以实施的。本研究使用神经进化来发现治疗的最佳策略。基于30天的治疗和单个患者的测量数据集，首先训练了随机森林来预测未来的血糖水平。然后通过进化了一个神经网络来指定碳水化合物摄入量、基础注射水平和突发注射。进化发现了一个帕累托前沿，减少了与目标值的偏差。

    Diabetes, a chronic condition that impairs how the body turns food into energy, i.e. blood glucose, affects 38 million people in the US alone. The standard treatment is to supplement carbohydrate intake with an artificial pancreas, i.e. a continuous insulin pump (basal shots), as well as occasional insulin injections (bolus shots). The goal of the treatment is to keep blood glucose at the center of an acceptable range, as measured through a continuous glucose meter. A secondary goal is to minimize injections, which are unpleasant and difficult for some patients to implement. In this study, neuroevolution was used to discover an optimal strategy for the treatment. Based on a dataset of 30 days of treatment and measurements of a single patient, a random forest was first trained to predict future glucose levels. A neural network was then evolved to prescribe carbohydrates, basal pumping levels, and bolus injections. Evolution discovered a Pareto front that reduced deviation from the targe
    
[^3]: 增强神经网络对多样化$\ell_p$攻击的鲁棒性:鲁棒模态连接导向的对抗性防御

    Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks. (arXiv:2303.10225v1 [cs.AI])

    [http://arxiv.org/abs/2303.10225](http://arxiv.org/abs/2303.10225)

    本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。

    

    对抗性鲁棒性是衡量神经网络在推理阶段抵御对抗性攻击能力的关键概念。最近的研究表明，尽管使用的强化鲁棒性训练技术能够提高对一种类型的攻击的鲁棒性，但模型仍然容易受到多样化的$\ell_p$攻击。为了实现多样化的$\ell_p$鲁棒性，我们提出了一种新颖的鲁棒模态连接 (RMC) 导向的对抗性防御，它包含两个基于种群学习的学习阶段。第一个阶段，RMC，能够搜索两个预先训练模型之间的模型参数空间，并找到包含高鲁棒性点的路径以抵御多样化的$\ell_p$攻击。基于RMC的有效性，我们开发了第二个阶段，基于RMC的优化，其中RMC作为神经网络多样化$\ell_p$鲁棒性进一步增强的基本单元。为了提高计算效率，我们将学习与仅选择子集的对抗性示例相结合，这导致了一组较小的代表性对抗性示例，可用于增强神经网络对多样化$\ell_p$攻击的鲁棒性。

    Adversarial robustness is a key concept in measuring the ability of neural networks to defend against adversarial attacks during the inference phase. Recent studies have shown that despite the success of improving adversarial robustness against a single type of attack using robust training techniques, models are still vulnerable to diversified $\ell_p$ attacks. To achieve diversified $\ell_p$ robustness, we propose a novel robust mode connectivity (RMC)-oriented adversarial defense that contains two population-based learning phases. The first phase, RMC, is able to search the model parameter space between two pre-trained models and find a path containing points with high robustness against diversified $\ell_p$ attacks. In light of the effectiveness of RMC, we develop a second phase, RMC-based optimization, with RMC serving as the basic unit for further enhancement of neural network diversified $\ell_p$ robustness. To increase computational efficiency, we incorporate learning with a sel
    

