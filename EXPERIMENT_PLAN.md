# 实验总览

## 实验1: Dogs vs Cats 对比实验

### 1.1 数据增强对比实验
实验目的: 评估不同数据增强策略对模型的影响(可以再加其他增强方式)

- 1.1.1: CNN_BN + basic augmentation + AdamW✅
    配置: configs/{base,task/dogs_vs_cats,model/SimpleCNN,aug/basic,optim/adamw}.yaml
    结果：experiments/2025-11-01_14-48-27_dogs_vs_cats_cnn_bn
      
- 1.1.2: CNN_BN + light augmentation + AdamW✅
    配置: configs/{base,task/dogs_vs_cats,model/SimpleCNN,aug/light,optim/adamw}.yaml
    结果：experiments/2025-11-01_16-00-00_dogs_vs_cats_cnn_bn  
### 1.2 模型架构对比实验 (f)
实验目的: 评估不同CNN架构的性能差异

- 1.2.1: SimpleCNN + light aug + AdamW (baseline)✅
    配置: configs/{base,task/dogs_vs_cats,model/cnn_small,aug/light,optim/adamw}.yaml
    结果：experiments/2025-10-24_13-09-40_dogs_vs_cats_simple_cnn_binary
- 1.2.2: CNN_BN + light aug + AdamW ✅
    配置: configs/{base,task/dogs_vs_cats,model/cnn_bn,aug/light,optim/adamw}.yaml
    结果： experiments/2025-10-26_14-51-41_dogs_vs_cats_cnn_bn
- 1.2.2: VGG-16 + light aug + AdamW (VGG有5种架构，可选其一跑)✅
    配置: configs/{base,task/dogs_vs_cats,model/VGG,aug/light,optim/adamw}.yaml  
    结果：experiments/2025-11-01_10-38-32_dogs_vs_cats_vgg
- 1.2.3: ResNet18 + light aug + AdamW✅
    配置: configs/{base,task/dogs_vs_cats,model/resnet18,aug/light,optim/adamw}.yaml
    结果：experiments/2025-11-01_12-36-25_dogs_vs_cats_resnet18
- 1.2.4: ResNetXt + light aug + AdamW✅
    配置: configs/{base,task/dogs_vs_cats,model/resnetxt50,aug/light,optim/adamw}.yaml
    结果：experiments/2025-11-01_13-36-36_dogs_vs_cats_resnext50
- 1.2.5: DenseNet_SMALL+ light aug + AdamW✅     
    配置: configs/{base,task/dogs_vs_cats,model/densenet_small,aug/light,optim/adamw}.yaml
    结果：experiments/2025-11-03_05-55-53_dogs_vs_cats_densenet
- 1.2.6: DenseNet121+ light aug + AdamW ✅
    配置: configs/{base,task/dogs_vs_cats,model/densenet121,aug/light,optim/adamw}.yaml
    结果：experiments/2025-11-03_06-15-48_dogs_vs_cats_densenet


### 1.3 优化器和学习率对比实验 (c)
实验目的: 评估优化器和学习率对训练的影响
- 1.3.1: CNN_BN + AdamW (lr=0.01) ✅
         结果：experiments/2025-11-01_16-00-00_dogs_vs_cats_cnn_bn  
      
- 1.3.2: CNN_BN + SGD (lr=0.001, momentum=0.9)✅
    配置: configs/{base,task/dogs_vs_cats,model/SimpleCNN,aug/light,optim/sgd}.yaml
    结果：experiments/2025-11-01_17-47-32_dogs_vs_cats_cnn_bn
- 1.3.3: CNN_BN + SGD (lr=0.01, momentum=0.9)✅
    配置: configs/{base,task/dogs_vs_cats,model/SimpleCNN,aug/light,optim/sgd}.yaml
    结果：experiments/2025-11-02_04-45-10_dogs_vs_cats_cnn_bn
- 1.3.4: CNN_BN + AdamW (lr=0.001)✅
结果：Project2_Classification/experiments/2025-11-01_17-26-41_dogs_vs_cats_cnn_bn
- 1.3.4: CNN_BN + AdamW (lr=0.1)✅
         结果：experiments/2025-11-01_16-49-38_dogs_vs_cats_cnn_bn


## 实验2: 错误案例分析 (e)
- 2.1: 生成混淆矩阵
    src/inference/analyze_errors.py
      
- 2.2: 可视化10-20个错误分类案例
    分析: 模型在哪些情况下容易出错?
      
- 2.3: 可视化10-20个正确分类案例
    分析: 模型的强项是什么?
      
- 2.4: 分析低置信度预测
      统计: 置信度<80%的预测有多少? 准确率如何?

分析重点:
- 是否混淆相似姿态的猫狗?
- 背景复杂时是否影响判断?
- 部分遮挡的情况如何?
- 多只动物时是否出错?

## 实验3: CIFAR-10 基础训练 (g)

- 3.1: ResNet18 + CIFAR-10 + basic aug + AdamW
    配置: configs/{base,task/cifar10,model/resnet18,aug/basic,optim/adamw}.yaml
      
### 3.2: CIFAR-10 类别不平衡处理 (h)，测试三种处理方法:
- 3.2.1: 加权损失函数 (Class Weights)。对少数类赋予更高的loss权重
         ResNet18 + 不平衡CIFAR-10 + Class Weighted CE Loss
      
- 3.2.2: 过采样 (Oversampling)，在训练时对少数类进行重复采样
         ResNet18 + 不平衡CIFAR-10 + WeightedRandomSampler
    
- 3.2.3: Focal Loss，降低易分类样本的权重，专注于困难样本
         ResNet18 + 不平衡CIFAR-10 + Focal Loss
         参考: https://arxiv.org/abs/1708.02002


对比指标:
- 整体准确率 (不够)
- Per-class准确率
- F1-score 
- 混淆矩阵 (看是否偏向多数类)


## 实验总结与报告

a) 数据集描述 ✓
   - Dogs vs Cats: 20K训练, 5K验证, 500测试
   - 预处理: Resize(224), ToTensor, Normalize(ImageNet统计量)
   - 增强: RandomResizedCrop, RandomHorizontalFlip

b) 模型架构 ✓
   - VGG-16: 13层卷积 + 3层全连接
   - ResNet18: 残差连接
   - 输入: 224x224x3
   - 输出: 2 (Dogs vs Cats) 或 10 (CIFAR-10)
   - Loss: CrossEntropyLoss / Binary CrossEntropy

c) 参数调优 → 实验3.1-3.3
   - 学习率: 0.01 vs 0.001 vs 0.1
   - 优化器: AdamW vs SGD

d) 验证准确率 + 提交文件
   - 使用最佳checkpoint
   - 生成predictions.csv → submission.csv

e) 错误案例分析 → 实验阶段2
   - 混淆矩阵
   - 可视化错误/正确案例
   - 讨论模型strengths和weaknesses

f) 模型和数据处理影响 
   - 数据增强: basic < light
   - 模型: SimpleCNN  VGG  ResNet
   - 交叉对比结果

g) CIFAR-10多分类 
   - 10类分类问题
   - 与Dogs vs Cats对比

h) CIFAR-10不平衡处理 
   - 方法1: Class Weights
   - 方法2: Oversampling
   - 方法3: Focal Loss
   - 对比per-class准确率

