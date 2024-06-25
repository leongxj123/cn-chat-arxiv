# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient LLM inference solution on Intel GPU.](http://arxiv.org/abs/2401.05391) | 本文提出了一种在Intel GPU上高效的LLM推理解决方案，通过简化LLM解码层和引入分段KV缓存策略，实现了低延迟和高吞吐量。 |

# 详细

[^1]: 在Intel GPU上高效的LLM推理解决方案

    Efficient LLM inference solution on Intel GPU. (arXiv:2401.05391v1 [cs.AR])

    [http://arxiv.org/abs/2401.05391](http://arxiv.org/abs/2401.05391)

    本文提出了一种在Intel GPU上高效的LLM推理解决方案，通过简化LLM解码层和引入分段KV缓存策略，实现了低延迟和高吞吐量。

    

    基于Transformer的大型语言模型（LLM）在许多领域广泛应用，LLM推理的效率成为实际应用中的热门话题。然而，LLM通常在模型结构上设计复杂，具有大量操作，并以自回归模式进行推理，这使得设计一个高效的系统成为一项具有挑战性的任务。在本文中，我们提出了一种高效的LLM推理解决方案，具有低延迟和高吞吐量。首先，我们通过融合数据移动和逐元素操作简化了LLM解码层，以减少内存访问频率并降低系统延迟。我们还提出了一种分段KV缓存策略，将请求和响应令牌的键/值分别保存在不同的物理内存中，以实现有效的设备内存管理，有助于增大运行时批处理大小并提高系统吞吐量。我们设计了一个定制的Scaled-Dot-Product-Attention内核，以匹配我们的融合策略，基于分段KV缓存解决方案。我们实验证明，该解决方案在Intel GPU上实现了高效的LLM推理。

    Transformer based Large Language Models (LLMs) have been widely used in many fields, and the efficiency of LLM inference becomes hot topic in real applications. However, LLMs are usually complicatedly designed in model structure with massive operations and perform inference in the auto-regressive mode, making it a challenging task to design a system with high efficiency.  In this paper, we propose an efficient LLM inference solution with low latency and high throughput. Firstly, we simplify the LLM decoder layer by fusing data movement and element-wise operations to reduce the memory access frequency and lower system latency. We also propose a segment KV cache policy to keep key/value of the request and response tokens in separate physical memory for effective device memory management, helping enlarge the runtime batch size and improve system throughput. A customized Scaled-Dot-Product-Attention kernel is designed to match our fusion policy based on the segment KV cache solution. We im
    

