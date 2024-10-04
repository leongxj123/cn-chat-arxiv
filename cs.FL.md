# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transformers as Transducers](https://arxiv.org/abs/2404.02040) | 通过将变压器与有限传感器联系起来，我们发现它们可以表达令人惊讶的大类传感，进一步扩展了RASP，推出了新的变体，并展示了掩码平均困难注意变压器可以模拟S-RASP. |

# 详细

[^1]: 变压器作为传感器

    Transformers as Transducers

    [https://arxiv.org/abs/2404.02040](https://arxiv.org/abs/2404.02040)

    通过将变压器与有限传感器联系起来，我们发现它们可以表达令人惊讶的大类传感，进一步扩展了RASP，推出了新的变体，并展示了掩码平均困难注意变压器可以模拟S-RASP.

    

    我们通过将变压器与有限传感器联系起来，研究了变压器的序列到序列映射能力，并发现它们可以表达令人惊讶的大类传感。我们使用RASP的变体，这是一种旨在帮助人们“像变压器一样思考”的编程语言，作为中间表示。我们扩展了现有的布尔变体B-RASP到序列到序列函数，并展示了它确切计算了一阶有理函数（如字符串旋转）。然后，我们引入了两个新扩展。B-RASP[pos]允许在位置上进行计算（如复制字符串的前半部分），并包含所有一阶正则函数。S-RASP添加前缀和，可以进行额外的算术操作（如对字符串求平方），并包含所有一阶多正则函数。最后，我们展示了掩码平均困难注意变压器可以模拟S-RASP。我们结果的一个推论是n...

    arXiv:2404.02040v1 Announce Type: cross  Abstract: We study the sequence-to-sequence mapping capacity of transformers by relating them to finite transducers, and find that they can express surprisingly large classes of transductions. We do so using variants of RASP, a programming language designed to help people "think like transformers," as an intermediate representation. We extend the existing Boolean variant B-RASP to sequence-to-sequence functions and show that it computes exactly the first-order rational functions (such as string rotation). Then, we introduce two new extensions. B-RASP[pos] enables calculations on positions (such as copying the first half of a string) and contains all first-order regular functions. S-RASP adds prefix sum, which enables additional arithmetic operations (such as squaring a string) and contains all first-order polyregular functions. Finally, we show that masked average-hard attention transformers can simulate S-RASP. A corollary of our results is a n
    

