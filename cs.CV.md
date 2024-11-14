# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective](https://arxiv.org/abs/2403.18346) | 提出了一个因果框架用于解释多模态大型语言模型在视觉问答问题中的偏差，并引入了一个新的挑战性数据集MORE，同时提出两种减轻单模态偏差的策略。 |
| [^2] | [V-LoL: A Diagnostic Dataset for Visual Logical Learning.](http://arxiv.org/abs/2306.07743) | V-LoL是一个结合视觉和逻辑挑战的诊断数据集，其中包括了V-LoL-Trains，该数据集首次将复杂的视觉场景和灵活的逻辑推理任务结合起来，为研究广泛的视觉逻辑学习挑战提供了平台。 |

# 详细

[^1]: 在多模态大型语言模型中量化和减轻单模态偏差：因果关系视角

    Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective

    [https://arxiv.org/abs/2403.18346](https://arxiv.org/abs/2403.18346)

    提出了一个因果框架用于解释多模态大型语言模型在视觉问答问题中的偏差，并引入了一个新的挑战性数据集MORE，同时提出两种减轻单模态偏差的策略。

    

    大型语言模型（LLMs）的最新进展促进了多模态LLMs（MLLMs）的发展。尽管它们具有令人印象深刻的能力，但MLLMs通常过度依赖单模态偏差（例如语言偏差和视觉偏差），导致在复杂多模态任务中给出不正确答案。为了调查这个问题，我们提出了一个因果框架来解释视觉问答（VQA）问题中的偏差。在我们的框架内，我们设计了一个因果图来阐明MLLMs对VQA问题的预测，并通过深入的因果分析评估偏差的因果效果。受因果图的启发，我们引入了一个新颖的MORE数据集，包含12,000个VQA实例。该数据集旨在挑战MLLMs的能力，需要多跳推理和克服单模态偏差。此外，我们提出了两种策略来减轻单模态偏差并增强MLLMs的推理能力。

    arXiv:2403.18346v1 Announce Type: new  Abstract: Recent advancements in Large Language Models (LLMs) have facilitated the development of Multimodal LLMs (MLLMs). Despite their impressive capabilities, MLLMs often suffer from an over-reliance on unimodal biases (e.g., language bias and vision bias), leading to incorrect answers in complex multimodal tasks. To investigate this issue, we propose a causal framework to interpret the biases in Visual Question Answering (VQA) problems. Within our framework, we devise a causal graph to elucidate the predictions of MLLMs on VQA problems, and assess the causal effect of biases through an in-depth causal analysis. Motivated by the causal graph, we introduce a novel MORE dataset, consisting of 12,000 VQA instances. This dataset is designed to challenge MLLMs' abilities, necessitating multi-hop reasoning and the surmounting of unimodal biases. Furthermore, we propose two strategies to mitigate unimodal biases and enhance MLLMs' reasoning capabiliti
    
[^2]: V-LoL: 一种用于视觉逻辑学习的诊断数据集

    V-LoL: A Diagnostic Dataset for Visual Logical Learning. (arXiv:2306.07743v1 [cs.AI])

    [http://arxiv.org/abs/2306.07743](http://arxiv.org/abs/2306.07743)

    V-LoL是一个结合视觉和逻辑挑战的诊断数据集，其中包括了V-LoL-Trains，该数据集首次将复杂的视觉场景和灵活的逻辑推理任务结合起来，为研究广泛的视觉逻辑学习挑战提供了平台。

    

    尽管近期在视觉AI领域有了许多成功的进展，但仍存在不同的缺点；包括缺少精确的逻辑推理、抽象的概括能力以及理解复杂和嘈杂的场景等。不幸的是，现有的基准测试数据集并不能捕捉到这些方面中的多数。深度学习数据集关注视觉复杂数据但只有简单的视觉推理任务，归纳逻辑数据集包括复杂的逻辑学习任务，但是缺乏视觉的组成部分。为了解决这个问题，我们提出了视觉逻辑学习数据集V-LoL，它无缝地结合了视觉和逻辑的挑战。值得注意的是，我们首次推出了V-LoL的第一个实例，名为V-LoL-Trains，它是符号AI中一个经典基准测试的视觉呈现，即Michalski火车问题。通过在一个通用框架内结合复杂的视觉场景和灵活的逻辑推理任务，V-LoL-Trains为研究广泛的视觉逻辑学习挑战提供了平台。

    Despite the successes of recent developments in visual AI, different shortcomings still exist; from missing exact logical reasoning, to abstract generalization abilities, to understanding complex and noisy scenes. Unfortunately, existing benchmarks, were not designed to capture more than a few of these aspects. Whereas deep learning datasets focus on visually complex data but simple visual reasoning tasks, inductive logic datasets involve complex logical learning tasks, however, lack the visual component. To address this, we propose the visual logical learning dataset, V-LoL, that seamlessly combines visual and logical challenges. Notably, we introduce the first instantiation of V-LoL, V-LoL-Trains, -- a visual rendition of a classic benchmark in symbolic AI, the Michalski train problem. By incorporating intricate visual scenes and flexible logical reasoning tasks within a versatile framework, V-LoL-Trains provides a platform for investigating a wide range of visual logical learning ch
    

