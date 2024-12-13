# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions](https://arxiv.org/abs/2402.05541) | 本研究提出了一个新的框架——强化联邦学习（RFL），通过利用深度强化学习自适应地优化客户贡献的聚合过程，以增强模型鲁棒性和在非相同分布环境下参与者之间的公平性。 |
| [^2] | [OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation.](http://arxiv.org/abs/2310.02422) | OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。 |

# 详细

[^1]: 强化学习作为鲁棒和公平联邦学习的催化剂：解密客户贡献动力学

    Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions

    [https://arxiv.org/abs/2402.05541](https://arxiv.org/abs/2402.05541)

    本研究提出了一个新的框架——强化联邦学习（RFL），通过利用深度强化学习自适应地优化客户贡献的聚合过程，以增强模型鲁棒性和在非相同分布环境下参与者之间的公平性。

    

    最近在联邦学习（FL）方面的进展产生了模型，通过在多个分散的设备或系统上训练来保护用户隐私并保留本地数据样本。然而，这些策略经常忽视统计异质性和对敌对攻击的脆弱性所带来的困难，这些因素会降低模型的鲁棒性和公平性。个性化的FL策略可以通过调整模型来适应个别客户的特点，但往往忽视了服务器端聚合的脆弱性。为了解决这些问题，我们提出了强化联邦学习（RFL），这是一个利用深度强化学习来自适应优化聚合过程中客户贡献的新框架，从而增强恶意客户下的模型鲁棒性和参与者之间的公平性在非相同分布环境下。为了实现这一目标，我们提出了一种细致的方法，其中包括基于深度确定性策略梯度算法的协同训练，以优化客户贡献的过程。

    Recent advancements in federated learning (FL) have produced models that retain user privacy by training across multiple decentralized devices or systems holding local data samples. However, these strategies often neglect the inherent challenges of statistical heterogeneity and vulnerability to adversarial attacks, which can degrade model robustness and fairness. Personalized FL strategies offer some respite by adjusting models to fit individual client profiles, yet they tend to neglect server-side aggregation vulnerabilities. To address these issues, we propose Reinforcement Federated Learning (RFL), a novel framework that leverages deep reinforcement learning to adaptively optimize client contribution during aggregation, thereby enhancing both model robustness against malicious clients and fairness across participants under non-identically distributed settings. To achieve this goal, we propose a meticulous approach involving a Deep Deterministic Policy Gradient-based algorithm for co
    
[^2]: OneAdapt：通过反向传播实现深度学习应用的快速自适应

    OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation. (arXiv:2310.02422v1 [cs.LG])

    [http://arxiv.org/abs/2310.02422](http://arxiv.org/abs/2310.02422)

    OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。

    

    深度学习在流媒体数据的推断方面已经普及，如视频中的目标检测、LiDAR数据和音频波形中的文本提取。为了实现高推断准确性，这些应用通常需要大量的网络带宽来收集高保真数据，并且需要广泛的GPU资源来运行深度神经网络(DNN)。尽管通过优化配置参数（如视频分辨率和帧率）可以大大减少对网络带宽和GPU资源的需求，但目前的自适应技术无法同时满足三个要求：（i）以最小的额外GPU或带宽开销来自适应配置；（ii）基于数据对最终DNN的准确性的影响来达到接近最优的决策；（iii）针对一系列配置参数进行自适应。本文提出了OneAdapt，通过利用梯度上升策略来自适应配置参数，满足了这些要求。关键思想是充分利用DNN的不同

    Deep learning inference on streaming media data, such as object detection in video or LiDAR feeds and text extraction from audio waves, is now ubiquitous. To achieve high inference accuracy, these applications typically require significant network bandwidth to gather high-fidelity data and extensive GPU resources to run deep neural networks (DNNs). While the high demand for network bandwidth and GPU resources could be substantially reduced by optimally adapting the configuration knobs, such as video resolution and frame rate, current adaptation techniques fail to meet three requirements simultaneously: adapt configurations (i) with minimum extra GPU or bandwidth overhead; (ii) to reach near-optimal decisions based on how the data affects the final DNN's accuracy, and (iii) do so for a range of configuration knobs. This paper presents OneAdapt, which meets these requirements by leveraging a gradient-ascent strategy to adapt configuration knobs. The key idea is to embrace DNNs' different
    

