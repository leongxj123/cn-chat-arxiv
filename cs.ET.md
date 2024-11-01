# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Cryogenic Memristive Neural Decoder for Fault-tolerant Quantum Error Correction.](http://arxiv.org/abs/2307.09463) | 本研究报告了一种基于内存计算的神经解码器推理加速器的设计和性能分析，该解码器用于容错量子错误纠正，旨在最小化解码时间并确保解码方法的可扩展性。 |

# 详细

[^1]: 一种用于容错量子错误纠正的低温存储敏化神经解码器

    A Cryogenic Memristive Neural Decoder for Fault-tolerant Quantum Error Correction. (arXiv:2307.09463v1 [quant-ph])

    [http://arxiv.org/abs/2307.09463](http://arxiv.org/abs/2307.09463)

    本研究报告了一种基于内存计算的神经解码器推理加速器的设计和性能分析，该解码器用于容错量子错误纠正，旨在最小化解码时间并确保解码方法的可扩展性。

    

    量子错误纠正(QEC)的神经解码器依赖于神经网络来分类从错误纠正编码中提取的综合征，并找到合适的恢复操作员来保护逻辑信息免受错误影响。尽管神经解码器性能良好，但仍存在一些重要的实际要求，如将解码时间最小化以满足重复错误纠正方案中的综合征生成速度，以及确保解码方法的可扩展性随着编码距离的增加而增加。设计一个专用的集成电路以与量子处理器共同完成解码任务似乎是必要的，因为将信号从低温环境中引出并进行外部处理会导致不必要的延迟和最终的布线瓶颈。在这项工作中，我们报道了一种基于内存计算的神经解码器推理加速器的设计和性能分析。

    Neural decoders for quantum error correction (QEC) rely on neural networks to classify syndromes extracted from error correction codes and find appropriate recovery operators to protect logical information against errors. Despite the good performance of neural decoders, important practical requirements remain to be achieved, such as minimizing the decoding time to meet typical rates of syndrome generation in repeated error correction schemes, and ensuring the scalability of the decoding approach as the code distance increases. Designing a dedicated integrated circuit to perform the decoding task in co-integration with a quantum processor appears necessary to reach these decoding time and scalability requirements, as routing signals in and out of a cryogenic environment to be processed externally leads to unnecessary delays and an eventual wiring bottleneck. In this work, we report the design and performance analysis of a neural decoder inference accelerator based on an in-memory comput
    

