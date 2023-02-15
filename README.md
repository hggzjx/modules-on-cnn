Skip to content
Search or jump to…
Pull requests
Issues
Codespaces
Marketplace
Explore
 
@hggzjx 
hggzjx
/
modules-on-cnn
Public
Cannot fork because you own this repository and are not a member of any organizations.
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
modules-on-cnn/readme.txt
@hggzjx
hggzjx Add files via upload
Latest commit 71a2598 now
 History
 1 contributor
42 lines (23 sloc)  2.31 KB

这几天看了几篇CV语义分割的论文，写了一些围绕着CNN的提升模型性能的插件。在基于预训练微调的大环境下这些插件可以很容易地嵌入模型之中,提高网络提取特征的能力或者信息交互能力（感觉在孪生网络中有很大的发挥空间）。这里写的插件，在很多SOTA模型中会看到它们的影子，甚至就是这些插件的拼接组合。相较于很多基本都在为了play而去plug，结果加入后还不如baseline的Attention，这些插件的效果应该经得起推敲。不过归根结底这些都是trick，只能提升模型的上限，真正有意义的学术还是提升模型的下限。
我是调包侠，用的Keras接口写的，这些插件只要input的shape是（batch_size,height,width,channels）就行，还有一堆超参，调这些超参应该比网络的实现麻烦很多。

module：adaptively_spatial_feature_fusion（ASFF）
paper：Adaptively Spatial Feature Fusion Learning Spatial Fusion for Single-Shot Object Detection
https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1911.09516v1

module：atrous_spatial_pyramid_pooling（ASPP）
paper:DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Conv
https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1606.00915.pdf

module：non_local_block
paper:Non-local Neural Networks
https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.07971

module：squeeze_excite_block (SE)
paper:Squeeze-and-Excitation Networks
https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1709.01507.pdf

module：cbam_block
paper:CBAM: Convolutional Block Attention Module
https://link.zhihu.com/?target=https%3A//openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

module：ghost_block
paper:GhostNet: More Features from Cheap Operations
https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1911.11907.pdf

module：receptive_field_block
paper:Receptive Field Block Net for Accurate and Fast Object Detection
https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.07767











                                                                                                                                                                                                       
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
modules-on-cnn/readme.txt at main · hggzjx/modules-on-cnn
