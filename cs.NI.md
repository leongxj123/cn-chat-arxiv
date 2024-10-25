# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ServeFlow: A Fast-Slow Model Architecture for Network Traffic Analysis](https://arxiv.org/abs/2402.03694) | ServeFlow提出了一种快速-慢速模型架构，用于网络流量分析。通过精心选择收集数据包的数量和应用于不同流量的模型，ServeFlow实现了最小延迟、高服务率和高准确性之间的平衡。在测试中，ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！ |

# 详细

[^1]: ServeFlow：一种用于网络流量分析的快速-慢速模型架构

    ServeFlow: A Fast-Slow Model Architecture for Network Traffic Analysis

    [https://arxiv.org/abs/2402.03694](https://arxiv.org/abs/2402.03694)

    ServeFlow提出了一种快速-慢速模型架构，用于网络流量分析。通过精心选择收集数据包的数量和应用于不同流量的模型，ServeFlow实现了最小延迟、高服务率和高准确性之间的平衡。在测试中，ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！

    

    随着互联网的整合和流量的加密，网络流量分析越来越多地使用复杂的机器学习模型。然而，在高带宽网络中，流量往往比模型的推理速率更快。网络流量的时间性质限制了在其他高流量机器学习应用中使用的简单扩展方法。因此，本文提出了ServeFlow，这是一个针对网络流量分析任务的机器学习模型提供解决方案，它通过精心选择收集数据包的数量和应用于不同流量的模型，来实现最小延迟、高服务率和高准确性之间的平衡。我们发现在相同的任务上，不同模型的推理时间可以相差2.7倍到136.3倍，而中位数数据包等待时间通常比推理时间高6到8个数量级！ServeFlow能够在16ms内对76.3%的流量进行推理，这是中位数推理时间的40.5倍加速！

    Network traffic analysis increasingly uses complex machine learning models as the internet consolidates and traffic gets more encrypted. However, over high-bandwidth networks, flows can easily arrive faster than model inference rates. The temporal nature of network flows limits simple scale-out approaches leveraged in other high-traffic machine learning applications. Accordingly, this paper presents ServeFlow, a solution for machine-learning model serving aimed at network traffic analysis tasks, which carefully selects the number of packets to collect and the models to apply for individual flows to achieve a balance between minimal latency, high service rate, and high accuracy. We identify that on the same task, inference time across models can differ by 2.7x-136.3x, while the median inter-packet waiting time is often 6-8 orders of magnitude higher than the inference time! ServeFlow is able to make inferences on 76.3% flows in under 16ms, which is a speed-up of 40.5x on the median end-
    

