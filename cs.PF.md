# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring the Impact of In-Browser Deep Learning Inference on Quality of User Experience and Performance](https://arxiv.org/abs/2402.05981) | 本研究通过全面性能评估，探索了浏览器内深度学习推理对用户体验质量和性能的影响。研究发现，浏览器内推理存在严重的延迟问题，平均比原生推理方法慢16.9倍。为了衡量这种影响，我们引入了新的指标：响应性，流畅度和推理准确性。 |
| [^2] | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | 该论文提出了一种无需调整的非对称2位量化KV缓存技术，以解决存储注意力键和值的内存需求增加和推断速度受限问题。 |

# 详细

[^1]: 探索浏览器内深度学习推理对用户体验质量和性能的影响

    Exploring the Impact of In-Browser Deep Learning Inference on Quality of User Experience and Performance

    [https://arxiv.org/abs/2402.05981](https://arxiv.org/abs/2402.05981)

    本研究通过全面性能评估，探索了浏览器内深度学习推理对用户体验质量和性能的影响。研究发现，浏览器内推理存在严重的延迟问题，平均比原生推理方法慢16.9倍。为了衡量这种影响，我们引入了新的指标：响应性，流畅度和推理准确性。

    

    深度学习越来越多地通过“浏览器内推理”这种方法整合到Web应用程序中，其中DL处理直接在Web浏览器中进行。然而，这种方法的实际性能及其对用户体验质量（QoE）的影响尚不为人所知。这种知识的空白需要新形式的QoE测量，超越传统的指标，如页面加载时间。为了解决这个问题，我们进行了浏览器内推理的首次全面性能评估。我们引入了新的指标：响应性，流畅度和推理准确性。我们的全面研究包括9个广泛使用的DL模型，并在50个常用的PC Web浏览器上进行了测试。研究结果显示，浏览器内推理存在严重的延迟问题：在CPU上平均比原生推理方法慢16.9倍，在GPU上慢4.9倍。这种延迟有几个因素导致，包括未充分使用的硬件指令集，固有的延迟等。

    Deep Learning (DL) is increasingly being integrated into Web applications through a method known as "in-browser inference", where the DL processes occur directly within Web browsers. However, the actual performance of this method and its effect on user experience quality (QoE) is not well-understood. This gap in knowledge necessitates new forms of QoE measurement, going beyond traditional metrics such as page load time. To address this, we conducted the first extensive performance evaluation of in-browser inference. We introduced new metrics for this purpose: responsiveness, smoothness, and inference accuracy.   Our thorough study included 9 widely-used DL models and tested them across 50 popular PC Web browsers. The findings show a significant latency issue with in-browser inference: it's on average 16.9 times slower on CPU and 4.9 times slower on GPU than native inference methods. Several factors contribute to this latency, including underused hardware instruction sets, inherent dela
    
[^2]: KIVI：一种无需调整的非对称2位量化KV缓存技术

    KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

    [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

    该论文提出了一种无需调整的非对称2位量化KV缓存技术，以解决存储注意力键和值的内存需求增加和推断速度受限问题。

    

    高效地为大型语言模型（LLMs）提供服务需要将许多请求批量处理以减少每个请求的成本。然而，存储注意力键和值以避免重新计算的键值（KV）缓存显著增加了内存需求，并成为速度和内存使用的新瓶颈。这种内存需求随着批处理大小和上下文长度的增加而增加。此外，推断速度受到KV缓存大小的限制，因为GPU的SRAM必须从主GPU内存中加载整个KV缓存以生成每个标记，导致计算核心在此过程中处于空闲状态。减小KV缓存大小的一个直接而有效的解决方案是量化，通过减少KV缓存所需的总字节数来实现。然而，目前缺乏对KV缓存元素分布进行深入研究以了解KV缓存量化的难度和限制。为了弥补这一空白，我们开展了一项全面的元素分布研究。。。

    Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, the key-value (KV) cache, which stores attention keys and values to avoid re-computations, significantly increases memory demands and becomes the new bottleneck in speed and memory usage. This memory demand increases with larger batch sizes and longer context lengths. Additionally, the inference speed is limited by the size of KV cache, as the GPU's SRAM must load the entire KV cache from the main GPU memory for each token generated, causing the computational core to be idle during this process. A straightforward and effective solution to reduce KV cache size is quantization, which decreases the total bytes taken by KV cache. However, there is a lack of in-depth studies that explore the element distribution of KV cache to understand the hardness and limitation of KV cache quantization. To fill the gap, we conducted a comprehensive study on the element distribut
    

