# Web Crawler Detect
# 项目报告已上传至report文件夹
## 说明
- 数据文件全部放到data目录下
- 除GenDataset.py以外其他代码文件可以不使用
  - CrawlerDetect为爬虫检测逻辑,我没有修改,作为模块被引用,本身没有调用
  - SessionCreate为会话划分,使用标准方案,可以设置会话超时时间
  - SessionProcess处理基本的爬虫检测以及一下附属功能
- 基础数据集为`sessions_detected.json`和`sessions_test_detected.json`
  - 分别来自两个不同的数据来源
  - 进行了session划分和基本的bot检查
    - 基本bot检查使用UA和robots.txt相关检测
- GenDataset.py生成相应的数据集
  - 参数为
    - input_file: 输入文件,两个基本数据集之一
    - output_file: 输出完整bot检测后的数据集
    - stat_file: 输出统计信息
    - min_request: 单个session中最少的请求数,少于该数字的不会写入output_file
    - threshold: bot检测的参数,越大越严格,越小越宽松,推荐范围为1~1.6
    - pos为保存路径,建议为文件夹dataset/,需要手动创建,否则写入失败
- 如有需要可以手动划分验证集或者融合两个数据集进行重划分,划分一个验证集有利于训练时调参
  - 验证集用于调整参数,达到模型最佳训练效果
  - 测试集用于检测模型泛化能力
- stat_file输出受到threshold影响,但不受min_request影响,统计信息包含所有session
  - 最后的字段为min_request=0的占比,不具有参考性,请使用excel表格选择参数
  - excel表格采用的是文件链接,但因为只能使用绝对路径,因此需要修改数量来源
  - 在保证stat_file基本字段不变的情况下本人的excel表格都可以正常使用
  - 修改链接的方式是:
    - 打开文件
    - 点击数据->查询和连接
    - 右边会出现一个侧栏,双击其中一个链接进入Power Query编辑器
    - 点击主页->数据源设置
    - 右键文件,选择更改源,然后选择目标json文件
    - 确定后关闭,点击刷新预览,然后点关闭并上载
    - 后续有数据更新之间点击刷新即可

## 内存管理
本人的台式机是32G内存,所以完整保存全部数据没有什么问题,
如果发现读取的时候爆内存了,windows可能会把内存里的数据转储到硬盘里,后续再读取
这会导致严重的性能问题
如果无法全量保存请考虑使用append模式逐个写入,统计代码对sessions这个变量没有依赖性
默认读取方式就是ijson,因此不会读取时就爆内存