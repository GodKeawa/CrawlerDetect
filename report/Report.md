# 1.数据集构建
## 1.1 数据来源
### 1.1.1 公开 HTTP Server 日志
- 来自数据集网站 Kaggle，数据量大，但是没有标注，也没有前人使用
	-  [Web Server Access Logs | Kaggle](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs)
- 来自各大论文使用的数据集，数据量大，略少于前者，没有公开的标注，但有各论文标注的爬虫比例，数据集为了隐私保护将一部分信息转成了 Hash 值
	- [Web robot detection - Server logs](https://zenodo.org/records/3477932)

优势在于均为真实访问日志，至少有大量搜索引擎的爬虫行为可以直接标注和分析

### 1.1.2 大模型生成日志
尝试找到一个好的 prompt 去引导大模型生成单个 session 的访问日志记录，但是有诸多问题：
- 时间的随机性无法正确生成
	- 爬虫往往固定间隔，即使要求提供一定的随机性也无法完全解决
- 无法形成对网站结构的了解
	- 一个网站的页面包含很多 reference，即使在提示中详细描述网站的结构，也无法在生成的内容中得到令人满意的访问流
- 不方便控制批处理
	- 使用 api 或者 ollama 本地部署都面临着上下文长度和生成内容解析的问题，LLM无法真正做到只生成日志，总喜欢多嘴，最多只能控制模型生成的日志包裹在两个标志中

### 1.1.3 自建网站进行生成
访问量过小，爬虫多样性无法模拟

我们最终选择使用公开的 HTTP Server 日志作为数据集，并参考论文的处理方式进行处理，尽量达到准确的标注和合理的 bot 比例

更好的数据来源是部分研究者主动构建的蜜罐网站，可以拿到纯爬虫的访问记录，但是没有找到公开的

## 1.2 数据处理
### 1.2.1 会话重建
使用 IP-UA 对进行用户区分，以 30 分钟为间隔进行会话划分
论文通用的处理方式
在 NAT 大量盛行的情况下对人类会话的区分率显著较低，公开数据集没有 user 标注或者 cookie 身份
对爬虫的区分率较高，尤其是搜索引擎爬虫一般不改变自身 IP 和 UA

### 1.2.1 基本检测
- UA 检测
	- 使用 Browscap 项目的公开模式
	- [Browser Capabilities Project](https://browscap.org/)
	- 使用几个开源匹配项目
- robots. txt 访问，sitemap 访问以及对禁止路径的访问

可以检测出全部搜索引擎的爬虫以及一些开源项目的爬虫 UA（大的开源爬虫项目大多在不配置 UA 的情况下使用项目的说明及项目地址作为 UA）
具有强区分能力，可以确保对爬虫的误报率很低，但是召回率不一定高

### 1.2.2 高级特征工程
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-06-21.jpg)

简单的模式分析方案，属于特征工程，实施由孙旭东完成，特征和权重为：
```python
score_weights = {  
    "high_speed": 0.6,              # 访问速度过快
    "regular_interval": 0.9,        # 访问间隔相近  
    "high_head_ratio": 0.5,         # 使用HEAD请求  
    "high_night_ratio": 0.3,        # 晚上访问比例高  
    "has_systematic_pattern": 0.4,  # 访问具有固定模式  
    "high_repetition": 0.5,         # 访问目标重复  
    "high_missing_referer": 0.7,    # 缺失referer  
    "high_bigram_repetition": 0.9,  # 访问的连续路径对重复 
    "high_404_ratio": 0.8,          # 高404  
    "high_4xx_ratio": 0.5,          # 高4xx  
    "low_size_variance": 0.7,       # 低大小方差 
    "low_avg_size": 0.4             # 低平均大小  
}
```
SVM 直接引用部分特征

### 1.2.3 参数选择
使用 Excel 动态数据链接实现自动打表，只需要选择格式相同的数据来源就可以保证函数正常运转
使用 Origin 进行一些作图可视化工作

### 1.2.4 实施时的数据处理方案
参数根据自己的算法需要选择
由于数据集显然是不平衡的，因此进行了两种方案，一种是调整检测机制，让数据集相对更偏向平衡，一种是直接控制采样，降低人类会话的采样率
- 可以在数据集上进行处理，随机抽取一定比例的人类会话样本，使得数据集基本平衡
- 可以在训练时进行处理，对不同类型数据调整采样率

由于有两个数据集，我的处理方式是将二者融合并打乱，再进行重划分
当数据量过大，导致训练时间过长时进行简单抽样

# 2. 模型构建
## 2.1 关于模型类型的选取
### 2.1.1 特征工程类模型
#### 监督学习模型
使用会话提取出的特征，线性嵌入到状态空间中，所有特征排列组成特征向量，在特征空间中寻找划分
- SVM 模型
- 逻辑回归模型

#### 非监督学习模型
同样提取特征，线性嵌入到状态空间，但是不尝试寻找划分，而是直接寻找聚类
- 聚类模型
- 异常检测模型

### 2.1.2 访问序列模型
通过对请求进行嵌入或者利用请求自身的可量化数值建立序列，使用模型发现序列中的隐藏特征
- 卷积神经网络
	- 直接通过卷积核捕捉序列特征
- Transformer 模型
	- 通过注意力获取前文信息

### 2.1.3 访问拓扑图模型
通过对网站的 sitemap 进行图构建，将会话中的访问路径按照时间顺序嵌入到网站的图中，通过捕捉访问有向图（也可以有边权或者点权）的特征进行分类
- 图神经网络

## 2.2 模型的实现
### 2.2.1 SVM 模型
直接使用 scikit-learn 中提供的 SVM，用 pytorch 搭建简单框架进行 GPU 加速训练

### 2.2.2 访问序列模型
参考论文的结构进行构建：
- 请求嵌入使用跳元模型进行非监督训练
	- 主要建模参考是访问路径，额外增加了请求方式和返回值
		- 其他可量化参数直接自行构建序列并进行通道融合
	- 窗口大小选择为 5，窗口构建选择步幅为 2，在最大化嵌入准度的条件下降低嵌入复杂度

- 主体使用卷积神经网络的加强版——残差神经网络
	- 使用单通道内的注意力机制实现特征的权重分配
	- 使用通道间的注意力机制进行特征融合

#### 跳元模型
主要原理便是通过词嵌入，尝试让中心词预测出周围词的概率最大（损失函数为预测正确概率的相反数）
希望通过这样的词嵌入把握网站的结构，因为主要的嵌入依据是请求的访问路径
使用特征字典作为 tokenizer，虽然可以用 LLM 的 tokenizer，但是因为一个网站的路径空间实际上不是很大，而文件名又非常混乱（大量Hash 值或者 Timestamp），使用 tokenizer 复杂度高，效果也不一定好，对于 hash 值的 token 化往往会让一个文件名变成非常多 token，不符合原本的网站结构
- 对路径进行字典 token 化，同时对文件（最后一个路径）进行全量字典
- HTTP 请求类型和状态码也可以简单用字典 token 化

全量训练，因为是上游模型，也不需要监督，最后达到 loss 基本低于 0.15，即预测正确的概率大于 85%

#### 访问序列模型
参考论文的模型架构，做出少量改进
##### ResNet
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-08-57.jpg)
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-09-10.jpg)
```python
class SNNBlock(nn.Module):  
    """残差网络块，使用1x1卷积实现残差连接"""  
  
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):  
        super(SNNBlock, self).__init__()  
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,  
                               padding=(kernel_size - 1) // 2, stride=stride)  
        self.bn1 = nn.BatchNorm1d(out_channels)  
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,  
                               padding=(kernel_size - 1) // 2, stride=stride)  
        self.bn2 = nn.BatchNorm1d(out_channels)  
  
        # 1x1卷积实现残差连接（特别是当输入输出通道数不同时）  
        self.shortcut = nn.Sequential()  
        if in_channels != out_channels:  
            self.shortcut = nn.Sequential(  
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),  
                nn.BatchNorm1d(out_channels)  
            )  
  
    def forward(self, x):  
        residual = self.shortcut(x)  	# 残差
  
        out = self.conv1(x)  			# 卷积
        out = self.bn1(out)  
        out = F.relu(out)  				# ReLU
  
        out = self.conv2(out)  			# 卷积
        out = self.bn2(out)  
  
        out += residual  				# 残差连接
        out = F.relu(out)  				# ReLU
  
        return out
```

##### 通道注意力机制
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_18-58-42.jpg)
```python
class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_features):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享MLP
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, in_features)

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()

        # 平均池化分支
        avg_y = self.avg_pool(x).squeeze(-1)
        avg_y = self.fc1(avg_y)
        avg_y = torch.tanh(avg_y)
        avg_y = self.fc2(avg_y)

        # 最大池化分支
        max_y = self.max_pool(x).squeeze(-1)
        max_y = self.fc1(max_y)
        max_y = torch.tanh(max_y)
        max_y = self.fc2(max_y)

        # 合并两个分支
        y = avg_y + max_y
        
        # 计算注意力权重
        attention = torch.softmax(y, dim=1).unsqueeze(-1)

        # 应用注意力
        out = x * attention

        return out
```
- 给通道进行加权
	- 对于请求嵌入的 32 位通道，每个通道可能隐藏了不同的特征，即使未隐藏也可以得到基本相同的权重
	- 对于请求时间和大小嵌入的 4 位通道，每个通道显然是不同的特征，可以进行权重分配

##### 基于注意力的通道融合
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-09-38.jpg)
```python
class AttentionFusion(nn.Module):  
    """基于注意力机制的特征融合模块"""  
  
    def __init__(self, feature_dim):  
        super(AttentionFusion, self).__init__()  
  
        # 定义Wqk权重矩阵，用于特征变换  
        self.W_qk = nn.Linear(feature_dim, feature_dim)  
  
        # 定义Wv权重矩阵，用于计算注意力系数  
        self.W_v = nn.Linear(feature_dim, 1)  
  
    def forward(self, features_list):  
        # 计算每个特征的注意力系数  
        attention_scores = []  
  
        for feature in features_list:  
            # 计算 a(Xi) = Wv·tanh(Wqk·Xi)            
            transformed = torch.tanh(self.W_qk(feature))  
            score = self.W_v(transformed)  
            attention_scores.append(score)  
  
        # 拼接所有注意力系数  
        attention_scores = torch.cat(attention_scores, dim=1)  
  
        # 计算注意力权重 α(Xi) = softmax(a(Xi))        
        attention_weights = F.softmax(attention_scores, dim=1)  
  
        # 加权平均  
        fused_feature = torch.zeros_like(features_list[0])  
        for i, feature in enumerate(features_list):  
            weight = attention_weights[:, i:i + 1]  
            fused_feature += feature * weight  
  
        return fused_feature
```

##### 整体结构
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-10-04.jpg)
```python
class SNNCD(nn.Module):  
    """基于一维卷积神经网络的爬虫检测模型"""  
  
    def __init__(self, time_length_channels=4, embed_channels=32,  
                 seq_length=256, hidden_channels=8, num_blocks=8):  
        super(SNNCD, self).__init__()  
  
        # 时间长度分支  
        self.time_length_branch = nn.Sequential(  
            SNNBlock(time_length_channels, hidden_channels, kernel_size=7),  
            *[SNNBlock(hidden_channels, hidden_channels, kernel_size=7)  
              for _ in range(num_blocks)]  
        )  
  
        # 请求嵌入分支  
        self.embed_branch = nn.Sequential(  
            SNNBlock(embed_channels, hidden_channels, kernel_size=7),  
            *[SNNBlock(hidden_channels, hidden_channels, kernel_size=7)  
              for _ in range(num_blocks)]  
        )  
  
        # 自适应平均池化  
        self.avg_pool = nn.AdaptiveAvgPool1d(64)  
  
        # 通道注意力机制  
        self.channel_attention = ChannelAttention(hidden_channels)  
  
        # 注意力融合模块  
        self.attention_fusion = AttentionFusion(64)  
  
        # 全连接分类层  
        self.classifier = nn.Sequential(  
            nn.Linear(64, 128),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(128, 16),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(16, 2)  
        )  
  
    def forward(self, time_length, embeddings):  
        # 时间长度分支  
        x1 = self.time_length_branch(time_length)  # [B, hidden, seq_len]  
        x1 = self.channel_attention(x1)  # 应用通道注意力  
        x1 = self.avg_pool(x1)  # [B, hidden, 64]  
        x1 = x1.mean(dim=1)  # [B, 64]  
  
        # 请求嵌入分支  
        x2 = self.embed_branch(embeddings)  # [B, hidden, seq_len]  
        x2 = self.channel_attention(x2)  # 应用通道注意力  
        x2 = self.avg_pool(x2)  # [B, hidden, 64]  
        x2 = x2.mean(dim=1)  # [B, 64]  
  
        # 使用注意力融合  
        x = self.attention_fusion([x1, x2])  # [B, 64]  
  
        # 分类  
        logits = self.classifier(x)  
  
        return logits
```

# 3. 模型训练
## 3.0 训练框架
训练框架整体使用标准的模型训练框架结构  
数据集划分为训练集，验证集和测试集，固定比例为 7：1.5：1.5  
训练时只用训练集进行训练，使用验证集判断训练程度，选取拟合程度和泛化能力最优的模型进行保存  
测试集专门用来进行测试

一些额外功能：
- 数据集 cache，将最终计算得到的数据集使用 pickle 保存，因为全是数组和内置数据类型，所有可以直接 pickle save 和 load（虽然 pickle 被认为是不安全的）
	- 考虑加载的数据集，如果发现有 cache 则直接读取 cache，可以减少 parse 的消耗，没有则进行 parse 和预处理，并保存到 cache 文件夹
- 过采样
	- 使用数据集直接实现，将训练集先分成两类，然后选择过采样率，将少的一类放大固定倍数，再融合打乱
- 详细的评测函数，保存所有预测概率和标签

## 3.1 跳元模型训练
进行数据量和参数的尝试，让训练时间相对可以接受但又不至于嵌入效果太差
```log
C:\Users\GodKe\.conda\envs\CudaEnv\python.exe D:\DevelopFolders\PyCharm\CrawlerDetect\src\SkipGram.py  
使用设备: cuda  
加载了 65462 个有效会话  
构建词汇表: 100%|██████████| 65462/65462 [00:27<00:00, 2424.27it/s]  
词汇表大小: 113865  
准备训练数据: 100%|██████████| 65462/65462 [00:46<00:00, 1393.70it/s]  
生成了 60257448 个训练样本  
Epoch 1/3: 100%|██████████| 941523/941523 [1:09:49<00:00, 224.73it/s, loss=0.1553]  
Epoch 2/3: 100%|██████████| 941523/941523 [1:11:12<00:00, 220.38it/s, loss=0.1451]  
Epoch 3/3: 100%|██████████| 941523/941523 [1:05:53<00:00, 238.14it/s, loss=0.1451]  
模型训练完成并保存到 models_balanced  
  
进程已结束，退出代码为 0
```

一共训练了三个词汇表：
- 一个是简单抽样，最少请求数 10，threshold 为 1
- 一个是平衡抽样，参数同上
- 一个是严格并平衡调整抽样，最少请求数 10，threshold 为 1.8

## 3.2 序列模型训练
### 3.2.0 关于评价指标和参数选择
#### 评价指标
通用四大分类模型指标  
[一文看懂机器学习指标：准确率、精准率、召回率、F1、ROC曲线、AUC曲线 - 知乎](https://zhuanlan.zhihu.com/p/93107394)  
ROC 曲线和 AUC  
[一文彻底理解机器学习 ROC-AUC 指标-CSDN博客](https://blog.csdn.net/sdgfafg_25/article/details/139621316)
-  什么是 ROC 曲线
	- ROC 曲线，即接收者操作特征曲线，ROC 曲线产生于第二次世界大战期间，最早用在信号检测领域，侦测战场上的敌军载具（飞机、船舰）。现在是是用来评价二分类模型性能的常用图形工具。它通过显示真阳性率（True Positive Rate，简称 TPR）与假阳性率（False Positive Rate，简称 FPR）之间的权衡来帮助我们理解模型的分类能力。
- 什么是 AUC
	- AUC，即曲线下面积（Area Under Curve），是 ROC 曲线下面积的一个数值表示。它提供了一个定量的指标，用来衡量分类模型的整体表现。AUC 值范围从 0 到 1，值越大表示模型性能越好。

#### 参数根据
在训练框架中本人增加了一个自主调整学习率的机制，机制由 pytorch. optim 直接提供  
内部实现细节未知，但整体逻辑就是先用大学习率，防止落入小的局部极值，在进入一个确定的凸区域后逐步缩小学习率，尽量接近极值  

其他参数和特性部分参考论文的一些探索，直接使用论文给出的参数，另一些由我主动测试

### 3.2.1 使用简单抽样数据集进行训练
```log
分类报告:  
              	precision    recall  f1-score   support  
      human      	0.93      0.97      0.95      5302  
         bot       	0.83      0.66      0.74      1134  
  	accuracy                            0.92      6436   
  	macro avg       0.88      0.82      0.84      6436
  	weighted avg    0.91      0.92      0.91      6436  
  
  
混淆矩阵:  
[[5145  157]  
 [ 382  752]]  
ROC AUC: 0.9423  
平均精度分数: 0.8394
```
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/roc_curve.png)
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/pr_curve.png)

### 3.2.2 使用平衡抽样数据集进行训练
```log
分类报告:  
   				   precision    recall  f1-score   support  
       human       		0.88      0.89      0.89      5500         
       bot         		0.88      0.88      0.88      5253  
       accuracy                        		0.88     10753
       macro avg   		0.88      0.88      0.88     10753
       weighted avg    	0.88      0.88      0.88     10753  
  
  
混淆矩阵:  
[[4893  607]  
 [ 636 4617]]  
ROC AUC: 0.9539  
平均精度分数: 0.9545
```
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/roc_curve2.png)
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/pr_curve2.png)
### 3.2.3 使用严格且平衡调整抽样数据集进行训练
```log
分类报告:
              precision    recall  f1-score   support

       human       0.94      0.98      0.96      6177
         bot       0.96      0.87      0.91      2857

    accuracy                           0.95      9034
   macro avg       0.95      0.93      0.94      9034
weighted avg       0.95      0.95      0.95      9034

混淆矩阵:
[[6070  107]
 [ 370 2487]]

ROC AUC: 0.9806
平均精度分数: 0.9701
```
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/roc_curve1.png)
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/pr_curve1.png)

### 3.2.4 额外参数改变的效果
经过一些基本验证，可以确定的是：
- 调整判定的严格程度，即逐渐接近几乎只由 UA 鉴别的标签
	- 越严格模型训练效果越好
	- 推测为搜索引擎爬虫的访问模式较为统一和单一
- 调整数据集平衡性及采样率
	- 平衡数据集或采样率均有利于 AUC，即提高整体性能，但准确率小幅度下降
- 调整模型的大小（加大通道数和卷积层数）
	- 加大药量并不会显著提高最终效果，甚至有过拟合的倾向
- 调整卷积核大小
	- 效果不明显，几乎也不作为参数

### 3.2.5 最终结果
参考论文结果  
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_20-02-14.jpg)  
相对最优结果  
```log
分类报告:
              precision    recall  f1-score   support

       human       0.95      0.98      0.96      6177
         bot       0.94      0.89      0.91      2857

    accuracy                           0.95      9034
   macro avg       0.95      0.93      0.94      9034
weighted avg       0.95      0.95      0.95      9034


混淆矩阵:
[[6023  154]
 [ 328 2529]]

ROC AUC: 0.9806
平均精度分数: 0.9703
```
可以看出复现程度已经接近原论文
![](https://raw.githubusercontent.com/GodKeawa/CrawlerDetect/refs/heads/master/report/img/PixPin_2025-04-28_22-10-51.jpg)


## 3.3 SVM 训练
SVM 面临一个问题，即打标签时使用的特征已经几乎穷尽了所有 SVM 可以使用的特征，因此训练 SVM 将会部分趋近于学习打标签函数的特性，这导致 SVM 可以超神也可以超鬼

选择大部分重要特征训练 SVM，各种数值都可以直接达到序列模型的水平，甚至部分约为 1，因为完全拟合一个函数对 SVM 来说是及其简单的，这样得到的成绩并不算数
```log
分类报告:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99     46828
           1       0.96      0.75      0.84      4652

    accuracy                           0.97     51480
   macro avg       0.97      0.87      0.91     51480
weighted avg       0.97      0.97      0.97     51480


混淆矩阵:
[[46682   146]
 [ 1154  3498]]
 
AUC值: 0.9662
```
选择部分重要特征，整体 f1接近 0.94，但是不平衡，AUC 较低，下面给出只有时间特征训练得到的 SVM，单次训练，没有进行网格优化等
```log
分类报告:
              precision    recall  f1-score   support

           0       0.95      0.99      0.97     78047
           1       0.81      0.47      0.59      7752

    accuracy                           0.94     85799
   macro avg       0.88      0.73      0.78     85799
weighted avg       0.94      0.94      0.93     85799

AUC: 0.8230
```

选择少量重要特征和非重要特征，非常不平衡，f1 分数很高但无用，AUC 很低
因为时间有限就不进行各种参数调整了，本身也是一个有问题的设计

# 4. 结尾与一些讨论
关于核心难点：
- 会话重建
	- 只有一个会话内的序列才是真的序列
- 请求嵌入/如何获取访问序列的特征
	- LogLLM 
	- 网站文件大量 hash 值命名怎么办，query 请求一堆逆天参数怎么办
- Cost
	- 大道至简的 SVM
	- 实时监测

