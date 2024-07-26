# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models.](http://arxiv.org/abs/2401.14351) | ServerlessLLM是一种用于大型语言模型的增强本地化无服务器推理系统，通过优化检查点加载、本地化推理和服务器分配来实现高效且低延迟的推理过程。 |

# 详细

[^1]: ServerlessLLM：增强本地化的用于大型语言模型的无服务器推理系统

    ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models. (arXiv:2401.14351v1 [cs.LG])

    [http://arxiv.org/abs/2401.14351](http://arxiv.org/abs/2401.14351)

    ServerlessLLM是一种用于大型语言模型的增强本地化无服务器推理系统，通过优化检查点加载、本地化推理和服务器分配来实现高效且低延迟的推理过程。

    

    本文介绍了ServerlessLLM，一种增强本地化的用于大型语言模型(LLM)的无服务器推理系统。ServerlessLLM利用GPU服务器上可用的存储和内存设备的大容量和带宽，从而减少昂贵的远程检查点下载并实现高效的检查点加载。ServerlessLLM通过三个主要贡献实现了这一目标：(i)通过一种新颖的加载优化检查点格式设计和高效的多级检查点加载系统实现快速LLM检查点加载；(ii)利用本地化推理和实时迁移，使ServerlessLLM能够在保持低延迟的同时有效地实现本地化驱动的服务器分配；(iii)本地化感知的服务器分配，使ServerlessLLM能够评估集群中每个服务器的状态，并有效地安排模型启动时间以充分利用本地检查点位置。我们进行了全面的实验，包括微基准测试和大规模语言模型评估，验证了ServerlessLLM的有效性和性能优势。

    This paper presents ServerlessLLM, a locality-enhanced serverless inference system for Large Language Models (LLMs). ServerlessLLM exploits the substantial capacity and bandwidth of storage and memory devices available on GPU servers, thereby reducing costly remote checkpoint downloads and achieving efficient checkpoint loading. ServerlessLLM achieves this through three main contributions: (i) fast LLM checkpoint loading via a novel loading-optimized checkpoint format design, coupled with an efficient multi-tier checkpoint loading system; (ii) locality-driven LLM inference with live migration, which allows ServerlessLLM to effectively achieve locality-driven server allocation while preserving the low latency of ongoing LLM inference; and (iii) locality-aware server allocation, enabling ServerlessLLM to evaluate the status of each server in a cluster and effectively schedule model startup time to capitalize on local checkpoint placement. Our comprehensive experiments, which include micr
    

