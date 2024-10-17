# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Do Humans Write Code? Large Models Do It the Same Way Too](https://arxiv.org/abs/2402.15729) | 大型语言模型在执行数值计算时经常出错，通过生成可执行代码来解决问题可以减少计算错误，但观察到当大型语言模型使用代码解决数学问题时，会生成更多不正确推理；为解决这一问题，提出了一种受人类编码实践启发的简单而高效方法Human-Think Language（HTL）。 |

# 详细

[^1]: 人类是如何编写代码的？大型模型也以同样的方式进行

    How Do Humans Write Code? Large Models Do It the Same Way Too

    [https://arxiv.org/abs/2402.15729](https://arxiv.org/abs/2402.15729)

    大型语言模型在执行数值计算时经常出错，通过生成可执行代码来解决问题可以减少计算错误，但观察到当大型语言模型使用代码解决数学问题时，会生成更多不正确推理；为解决这一问题，提出了一种受人类编码实践启发的简单而高效方法Human-Think Language（HTL）。

    

    大型语言模型（LLMs）在执行数值计算时经常出错。与传统的思维链推理相比，程序化思维方法涉及生成可执行代码来解决问题。通过执行这些代码，它可以获得更精确的结果。使用生成的可执行代码而不是自然语言可以减少计算错误。然而，我们观察到当LLMs使用代码解决数学问题时，他们往往生成比使用自然语言更多的不正确推理。为了解决这个问题，我们提出了Human-Think Language（HTL），这是一种受到人类编码实践启发的简单而高效的方法。该方法首先由模型生成用自然语言描述的解决问题方法，然后将其转换为代码，反映出人们在将逻辑以自然语言形式思考后再将其写成代码的过程。此外，它利用了P

    arXiv:2402.15729v1 Announce Type: new  Abstract: Large Language Models (LLMs) often make errors when performing numerical calculations. In contrast to traditional chain-of-thought reasoning, the program-of-thoughts approach involves generating executable code to solve problems. By executing this code, it achieves more precise results. Using generated executable code instead of natural language can reduce computational errors. However, we observe that when LLMs solve mathematical problems using code, they tend to generate more incorrect reasoning than when using natural language. To address this issue, we propose Human-Think Language (HTL), a straightforward yet highly efficient approach inspired by human coding practices. The approach first generates problem-solving methods described in the natural language by the model, then converts them into code, mirroring the process where people think through the logic in natural language before writing it as code. Additionally, it utilizes the P
    

