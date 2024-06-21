# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM4Decompile: Decompiling Binary Code with Large Language Models](https://arxiv.org/abs/2403.05286) | 发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。 |

# 详细

[^1]: LLM4Decompile：使用大型语言模型对二进制代码进行反编译

    LLM4Decompile: Decompiling Binary Code with Large Language Models

    [https://arxiv.org/abs/2403.05286](https://arxiv.org/abs/2403.05286)

    发布首批开放访问的反编译LLM，预训练在40亿个C源代码和汇编代码标记上，引入了第一个考虑重新编译性和重新执行性的反编译数据集。

    

    反编译旨在将编译代码恢复为可读性强的源代码，但在名称和结构等细节方面存在困难。大型语言模型（LLMs）在编程任务中显示出潜力，激发了它们在反编译中的应用。然而，目前尚无用于反编译的开源LLM。此外，现有的反编译评估系统主要考虑标记级准确性，而很大程度上忽略了代码的可执行性，这是任何程序最重要的特征。因此，我们发布了首批开放访问的反编译LLM，范围从10亿到330亿，预先训练了40亿个令牌的C源代码和相应的汇编代码。这些开源LLM可以作为该领域进一步发展的基线。为了确保实际程序评估，我们引入了Decompile-Eval，这是第一个考虑重新编译性和重新执行性的反编译数据集。该基准强调了评估的重要性。

    arXiv:2403.05286v1 Announce Type: cross  Abstract: Decompilation aims to restore compiled code to human-readable source code, but struggles with details like names and structure. Large language models (LLMs) show promise for programming tasks, motivating their application to decompilation. However, there does not exist any open-source LLM for decompilation. Moreover, existing decompilation evaluation systems mainly consider token-level accuracy and largely ignore code executability, which is the most important feature of any program. Therefore, we release the first open-access decompilation LLMs ranging from 1B to 33B pre-trained on 4 billion tokens of C source code and the corresponding assembly code. The open-source LLMs can serve as baselines for further development in the field. To ensure practical program evaluation, we introduce Decompile-Eval, the first dataset that considers re-compilability and re-executability for decompilation. The benchmark emphasizes the importance of eval
    

