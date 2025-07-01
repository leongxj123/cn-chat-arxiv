# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Value-Compressed Sparse Column (VCSC): Sparse Matrix Storage for Redundant Data.](http://arxiv.org/abs/2309.04355) | 值压缩的稀疏列（VCSC）是一种新的稀疏矩阵存储格式，能够利用高冗余性将数据进一步压缩，并在性能上没有显著的负面影响。通过增量编码和字节打包压缩索引数组，IVCSC实现了更大的存储空间节省。 |

# 详细

[^1]: 值压缩的稀疏列（VCSC）：冗余数据的稀疏矩阵存储

    Value-Compressed Sparse Column (VCSC): Sparse Matrix Storage for Redundant Data. (arXiv:2309.04355v1 [cs.DS])

    [http://arxiv.org/abs/2309.04355](http://arxiv.org/abs/2309.04355)

    值压缩的稀疏列（VCSC）是一种新的稀疏矩阵存储格式，能够利用高冗余性将数据进一步压缩，并在性能上没有显著的负面影响。通过增量编码和字节打包压缩索引数组，IVCSC实现了更大的存储空间节省。

    

    压缩的稀疏列（CSC）和坐标（COO）是稀疏矩阵的常用压缩格式。然而，CSC和COO都是通用格式，不能利用除稀疏性以外的数据特性，如数据冗余性。高度冗余的稀疏数据在许多机器学习应用中很常见，例如基因组学，在传统的稀疏存储格式下，这些数据通常太大无法进行内存计算。本文中，我们提出了两个扩展的CSC格式：值压缩的稀疏列（VCSC）和索引和值压缩的稀疏列（IVCSC）。VCSC利用列内的高冗余性，将数据进一步压缩了3倍以上，相比COO压缩了2.25倍，而性能特征没有显著的负面影响。IVCSC通过增量编码和字节打包压缩索引数组，使内存使用量比COO减少了10倍，比CSC减少了7.5倍。

    Compressed Sparse Column (CSC) and Coordinate (COO) are popular compression formats for sparse matrices. However, both CSC and COO are general purpose and cannot take advantage of any of the properties of the data other than sparsity, such as data redundancy. Highly redundant sparse data is common in many machine learning applications, such as genomics, and is often too large for in-core computation using conventional sparse storage formats. In this paper, we present two extensions to CSC: (1) Value-Compressed Sparse Column (VCSC) and (2) Index- and Value-Compressed Sparse Column (IVCSC). VCSC takes advantage of high redundancy within a column to further compress data up to 3-fold over COO and 2.25-fold over CSC, without significant negative impact to performance characteristics. IVCSC extends VCSC by compressing index arrays through delta encoding and byte-packing, achieving a 10-fold decrease in memory usage over COO and 7.5-fold decrease over CSC. Our benchmarks on simulated and rea
    

