# 城市租金预测：多元线性回归与正则化模型（机器学习期末大作业）

## 1. 项目简介
本项目以公开租房挂牌数据为基础，建立租金（月租金）预测回归模型。
在普通多元线性回归（OLS）基础上，引入 Ridge（L2）、Lasso（L1）与 ElasticNet
正则化方法，缓解多特征之间的相关性与多重共线性问题，并比较不同模型在预测误差
和特征权重上的差异。

## 2. 目录结构
```
大作业/
├── data/
│   └── manhattan.csv            # 原始公开数据集（StreetEasy 曼哈顿租房挂牌）
├── code/
│   ├── 01_data_exploration.py   # 数据清洗与探索性分析（EDA）
│   ├── 02_modeling.py           # 特征工程、模型训练、正则化对比与可视化
│   └── main.py                  # 统一运行入口
├── figures/                     # 实验生成的图表
├── output/
│   ├── cleaned_data.csv          # 清洗后数据
│   ├── metrics.csv              # 各模型评估指标
│   ├── coefficients.csv         # 各模型标准化特征权重
│   ├── predictions.csv           # 最优模型测试集预测明细
│   └── console_log.txt           # 运行控制台输出留档
└── README.md
```

## 3. 运行环境
- Python 3.10+（开发环境 Python 3.13）
- 依赖包：pandas、numpy、scikit-learn、matplotlib、seaborn、requests

安装依赖：
```
python -m pip install pandas numpy scikit-learn matplotlib seaborn requests
```

## 4. 运行步骤
1. 确认 `data/manhattan.csv` 已存在；
2. 在项目根目录执行：
```
python code/main.py
```
或分步执行：
```
python code/01_data_exploration.py
python code/02_modeling.py
```

## 5. 数据集说明
数据来自 Codecademy 公开课程镜像的 StreetEasy 曼哈顿租房挂牌数据
（GitHub: Codecademy/datasets，streeteasy/manhattan.csv），共 3539 条挂牌记录，
包含卧室数、卫生间数、面积、距地铁站步行时间、楼层、楼龄、是否免中介费、
各类配套设施以及所属片区等字段，目标变量为月租金（USD/month）。

## 6. 模型与评估指标
- 模型：OLS 多元线性回归、Ridge 岭回归、Lasso 套索回归、ElasticNet 弹性网络；
- 预处理：数值特征标准化、二值特征原样保留、片区类别独热编码；
- 调参：GridSearchCV + 5 折交叉验证；
- 指标：决定系数 R²、均方根误差 RMSE、平均绝对误差 MAE、平均绝对百分比误差 MAPE。
