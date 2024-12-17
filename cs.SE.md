# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DevBench: A Comprehensive Benchmark for Software Development](https://arxiv.org/abs/2403.08604) | DevBench是一个综合基准测试，评估大型语言模型在软件开发生命周期各个阶段的表现，并发现现有的模型在其中存在挑战。 |
| [^2] | [RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair](https://arxiv.org/abs/2312.15698) | 高效表示和微调适配器相结合的新型程序修复方法RepairLLaMA可为语言模型修复错误产生高效的适配器。 |

# 详细

[^1]: DevBench：软件开发的综合基准测试

    DevBench: A Comprehensive Benchmark for Software Development

    [https://arxiv.org/abs/2403.08604](https://arxiv.org/abs/2403.08604)

    DevBench是一个综合基准测试，评估大型语言模型在软件开发生命周期各个阶段的表现，并发现现有的模型在其中存在挑战。

    

    arXiv:2403.08604v1宣布类型：新的摘要：大型语言模型（LLMs）的最新进展显著提升了它们的编码能力。然而，现有的基准测试主要关注编程的简化或孤立方面，如单文件代码生成或存储库问题调试，未能全面衡量由真实世界编程活动提出的各种挑战的全谱。为此，我们提出了DevBench，一个综合基准测试，评估LLMs在软件开发生命周期的各个阶段，包括软件设计、环境设置、实现、验收测试和单元测试。DevBench具有各种编程语言和领域，高质量数据收集，并针对每个任务精心设计和验证的指标。实证研究表明，当前的LLMs，包括GPT-4-Turbo，无法解决DevBench提出的挑战。分析表明，模型难以理解

    arXiv:2403.08604v1 Announce Type: new  Abstract: Recent advancements in large language models (LLMs) have significantly enhanced their coding capabilities. However, existing benchmarks predominantly focused on simplified or isolated aspects of programming, such as single-file code generation or repository issue debugging, falling short of measuring the full spectrum of challenges raised by real-world programming activities. To this end, we propose DevBench, a comprehensive benchmark that evaluates LLMs across various stages of the software development lifecycle, including software design, environment setup, implementation, acceptance testing, and unit testing. DevBench features a wide range of programming languages and domains, high-quality data collection, and carefully designed and verified metrics for each task. Empirical studies show that current LLMs, including GPT-4-Turbo, fail to solve the challenges presented within DevBench. Analyses reveal that models struggle with understand
    
[^2]: RepairLLaMA：高效表示和微调适配器用于程序修复

    RepairLLaMA: Efficient Representations and Fine-Tuned Adapters for Program Repair

    [https://arxiv.org/abs/2312.15698](https://arxiv.org/abs/2312.15698)

    高效表示和微调适配器相结合的新型程序修复方法RepairLLaMA可为语言模型修复错误产生高效的适配器。

    

    自动程序修复（APR）随着大型语言模型（LLMs）的出现已有了显著发展。对于程序修复进行LLMs的微调是最近研究的一个新领域，有许多未被探索的维度。现有工作大多使用简单的代码表示对LLMs进行微调，并在能够微调更大型LLMs的能力方面存在根本性局限。为解决这个问题，我们提出了RepairLLaMA，一个结合了1）用于APR的代码表示和2）最先进的参数高效的LLM微调技术LoRA的新型程序修复方法。这使得RepairLLaMA产生了一个高效的“程序修复适配器”，用于使用语言模型修复错误。我们的实验证明了这两个概念的有效性。首先，使用具有程序修复特定代码表示的微调适配器使模型能够使用有意义的修复信号。其次，参数高效的微调有助于微调...

    arXiv:2312.15698v2 Announce Type: replace-cross  Abstract: Automated Program Repair (APR) has evolved significantly with the advent of Large Language Models (LLMs). Fine-tuning LLMs for program repair is a recent avenue of research, with many dimensions which have not been explored. Existing work mostly fine-tunes LLMs with naive code representations and is fundamentally limited in its ability to fine-tune larger LLMs. To address this problem, we propose RepairLLaMA, a novel program repair approach that combines 1) code representations for APR and 2) the state-of-the-art parameter-efficient LLM fine-tuning technique called LoRA. This results in RepairLLaMA producing a highly effective `program repair adapter' for fixing bugs with language models. Our experiments demonstrate the validity of both concepts. First, fine-tuning adapters with program repair specific code representations enables the model to use meaningful repair signals. Second, parameter-efficient fine-tuning helps fine-tun
    

