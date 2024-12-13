# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation.](http://arxiv.org/abs/2310.02422) | OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。 |

# 详细

[^1]: OneAdapt：通过反向传播实现深度学习应用的快速自适应

    OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation. (arXiv:2310.02422v1 [cs.LG])

    [http://arxiv.org/abs/2310.02422](http://arxiv.org/abs/2310.02422)

    OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。

    

    深度学习在流媒体数据的推断方面已经普及，如视频中的目标检测、LiDAR数据和音频波形中的文本提取。为了实现高推断准确性，这些应用通常需要大量的网络带宽来收集高保真数据，并且需要广泛的GPU资源来运行深度神经网络(DNN)。尽管通过优化配置参数（如视频分辨率和帧率）可以大大减少对网络带宽和GPU资源的需求，但目前的自适应技术无法同时满足三个要求：（i）以最小的额外GPU或带宽开销来自适应配置；（ii）基于数据对最终DNN的准确性的影响来达到接近最优的决策；（iii）针对一系列配置参数进行自适应。本文提出了OneAdapt，通过利用梯度上升策略来自适应配置参数，满足了这些要求。关键思想是充分利用DNN的不同

    Deep learning inference on streaming media data, such as object detection in video or LiDAR feeds and text extraction from audio waves, is now ubiquitous. To achieve high inference accuracy, these applications typically require significant network bandwidth to gather high-fidelity data and extensive GPU resources to run deep neural networks (DNNs). While the high demand for network bandwidth and GPU resources could be substantially reduced by optimally adapting the configuration knobs, such as video resolution and frame rate, current adaptation techniques fail to meet three requirements simultaneously: adapt configurations (i) with minimum extra GPU or bandwidth overhead; (ii) to reach near-optimal decisions based on how the data affects the final DNN's accuracy, and (iii) do so for a range of configuration knobs. This paper presents OneAdapt, which meets these requirements by leveraging a gradient-ascent strategy to adapt configuration knobs. The key idea is to embrace DNNs' different
    

