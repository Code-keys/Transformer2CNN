# Transformer2CNN
基于图像块的CNN架构，骨干+特征融合，逼近Transformer

<details open>
<summary>Info</summary> 
  
``` bash 
        基于可增大感受野的残差空洞卷积模块，提高了空间特征的利用率，在不
    降低模型实时性的同时显著提高了检测精度。
        其次，通过对多尺度的特征进行跨阶段特征融合，提出了 SCAM 模块，提
    高了特征融合效率与中大目标的检测精度。
        最后，通过 SDCM 检测头解耦模块，缓解了分类和回归之间    的互斥矛
    盾，进一步提高了检测性能。
```
</details>


<details open>
<summary>BackBone</summary> 

``` bash 
    基于图像块的特征提取
  ```
  
</details>
<details open>
<summary>Neck</summary> 

``` bash 
  基于图像块、注意力的特征融合
  ```
</details>
 
<details open>
<summary>Neck</summary> 

``` bash 
  基于 类间-解耦头设计 的细分类做法。
  ```
</details>
   

<details open>
<summary>citation</summary> 
- [1] xx, xxx, xx. 基于改进 YOLOV5s 的无人机图像实时目标检测[J]. 光电工程, 2022, 49(3): 210372.
</details>
