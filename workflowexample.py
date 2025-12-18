import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R # 实验记录管理器
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.utils import flatten_dict

from qlib.data import D # 基础行情数据服务对象
from qlib.contrib.report import analysis_position, analysis_model

# 完整实验工作流：train, predict, backtest

provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

stockpool = D.instruments(market='csi100')

benchmark = "SH000300" # 基准：沪深300指数

# 参数配置
# 数据处理器参数配置
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-31",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": stockpool, ### 记住是instruments而不是instrument
}

# 任务参数配置
task = {
    # 机器学习模型参数配置
    "model": {
        # 模型类
        "class": "LGBModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.gbdt",
        # 模型类超参数配置, 未写的采用默认值, 参数将传给模型类
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "early_stopping_rounds": 50, # 训练迭代提前停止条件
            "num_boost_round": 1000, # 最大训练迭代次数
        },
    },
    "dataset":{ # 因子数据集参数配置
        # 数据集类, 是Dataset with Data(H)andler的缩写
        "class": "DatasetH",
        # 数据集所在的模块
        "module_path": "qlib.data.dataset",
        # 数据集类的参数配置
        "kwargs": {
            "handler": {
                "class": "Alpha158",# 数据处理类,继承自DataHandlerLP
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments":{ # 数据集时段划分
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2014-12-31", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    },
}

# 实例化模型对象
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# 执行训练模型实验
with R.start( experiment_name="train"): # 注意：设置好实验名
    # 可选：记录task中的参数到运行记录下的params目录
    R.log_params(**flatten_dict(task))

    # 训练模型,得到训练好的model
    model.fit(dataset)

    # 训练好的模型以pkl文件形式保存到本次实验运行记录目录下的artifacts子目录
    R.save_objects(**{"trained_model.pkl": model})

    # 打印本次实验记录器信息，含记录器id，experiment_id等信息
    print('info', R.get_recorder().info)


# 执行预测实验
with R.start(experiment_name="predict"):

    # 当前实验的实验记录器
    predict_recorder = R.get_recorder()

    # 生成预测结果文件：pred.pkl，label.pkl存放在运行记录目录下的artifacts子目录
    # 本实验默认是站在t日结束时刻，预测t+2日收盘价相对t+1日的收益率，计算公式为 Ref($close, -2)/Ref($close, -1) - 1
    sig_rec = SignalRecord(model, dataset, predict_recorder)
    sig_rec.generate()

    print('info', R.get_recorder().info)

    # 生成预测结果分析文件，在artifacts\sig_analysis 目录生成ic.pkl, ric.pkl文件
    sigAna_rec = SigAnaRecord(predict_recorder)
    sigAna_rec.generate()

# 预测结果查询
# label_df = predict_recorder.load_object("label.pkl") # 这个pkl文件记录的是测试集未经数据预处理的原始标签值
label_df = dataset.prepare("test", col_set="label") # 测试集标签值，默认这是经过数据预处理的标签
label_df.columns = ['label'] # 修改列名LABEL0为label

pred_df = predict_recorder.load_object("pred.pkl") # 加载测试集预测结果到dataframe

print('label_df', label_df) # 预处理后的测试集标签值
print('pred_df', pred_df) # 测试集对标签的预测值，score就是预测值


# 信息系数：每天根据所有股票的预测值和标签值，计算出二者在该日的相关系数，即为该日信息系数
ic_df = predict_recorder.load_object("sig_analysis/ic.pkl")
print('ic_df', ic_df)
# 排序信息系数 rank ic: 每天根据所有股票的预测值的排名和标签值的排名，计算出二者在该日的排序相关系数，即为该日的排序信息系数
ric_df = predict_recorder.load_object("sig_analysis/ric.pkl")
print('ric_df', ric_df)

print('list_metrics', predict_recorder.list_metrics()) # 所有绩效指标
print('IC', predict_recorder.list_metrics()['IC']) # 平均IC：每日IC的均值
print('ICIR', predict_recorder.list_metrics()['ICIR']) # IC信息率：平均IC/每日IC标准差
print('Rank IC', predict_recorder.list_metrics()['Rank IC'])
print('Rank ICIR', predict_recorder.list_metrics()['Rank ICIR'])


# 预测绩效分析图
# 准备数据：测试集“预测值”和“标签值”对照表
pred_label_df = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)
print(pred_label_df)

# 信息系数ic和rank ic图（按天）
analysis_position.score_ic_graph(pred_label_df)
# ic图横坐标按天显示该日所有股票预测值和标签的相关系数
# 有时二者正相关，有时二者负相关。有时相关性很小(无法利用)

# 预测模型绩效图
analysis_model.model_performance_graph(pred_label_df)
# 设置参数
analysis_model.model_performance_graph(pred_label_df, N=6,
    graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
    rank=True, lag=1, reverse=False, show_notebook=True) # N分几组，lag:自相关图滞后期


## 模型特征重要性
# 得到特征重要性系列
feature_importance = model.get_feature_importance()
print(feature_importance)
# feature_importance.plot(figsize=(50, 10))

fea_expr, fea_name = dataset.handler.get_feature_config() # 获取特征表达式，特征名字
# 特征名，重要性值的对照字典
feature_importance = {fea_name[int(i.split('_')[1])] : v for i, v in feature_importance.items()}
print(feature_importance)


## 回测
# 回测所需参数配置
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": { # 回测策略相关超参数配置
        "class": "TopkDropoutStrategy", # 策略类名称
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            # "model": model, # 模型对象
            # "dataset": dataset, # 数据集
            "signal": (model, dataset), # 信号，也可以是pred_df，得到测试集的预测值score
            "topk": 50,
            "n_drop": 5,
            "only_tradable": True,
            "risk_degree": 0.95,
        },
    },
    "backtest":{ # 回测数据参数
        "start_time": "2017-01-01", # test集开始时间
        "end_time": "2020-08-01", # test集结束时间
        "account": 100000000,
        "benchmark": benchmark, # 基准
        "exchange_kwargs": {
            "freq": "day", # 使用日线数据
            "limit_threshold": 0.095, # 涨跌停板幅度
            "deal_price": "close", # 以收盘价成交
            "open_cost": 0.0005, # 开仓佣金费率
            "close_cost": 0.0015, # 平仓佣金费率
            "min_cost": 5, # 一笔交易的最小成本
            "impact_cost": 0.01, # 冲击成本费率，比如因滑点产生的冲击成本
            "trade_unit": 100, # 成交量必须为100股的整数倍
        },
    },
}

# 实验名“back test”
with R.start(experiment_name="backtest"):

    # 创建组合分析记录器，其中predict_recorder把预测值带进来
    # pa_rec是组合分析记录器portfolio analysis recorder的缩写
    pa_rec = PortAnaRecord(predict_recorder, port_analysis_config, "day")
    # 回测与分析：通过组合分析记录器，在测试集上执行策略回测，并记录分析结果到多个pkl文件
    # 保存到predict_recorder对应目录的子目录artifacts\portfolio_analysis
    # 而不是本次实验的目录下
    pa_rec.generate()

    print('predict_recorder.experiment_id', predict_recorder.experiment_id, 'predict_recorder.id', predict_recorder.id)
    print('info', R.get_recorder().info) # 本次实验信息

# 回测结果提取到df
indicators_normal_1day_df = predict_recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")
# indicators_normal_1day_obj_df = predict_recorder.load_object("portfolio_analysis/indicators_normal_1day_obj.pkl")
indicator_analysis_1day_df = predict_recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
port_analysis_1day_df = predict_recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
positions_normal_1day_df = predict_recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
report_normal_1day_df = predict_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")

print('indicator_analysis_1day_df', indicator_analysis_1day_df)
'''
https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py
pa is the price advantage in trade indicators
pos is the positive rate in trade indicators
ffr is the fulfill rate in trade indicators
'''
print('indicators_normal_1day_df', indicators_normal_1day_df)
# print('indicators_normal_1day_obj_df\n', indicators_normal_1day_obj_df)

from pprint import pprint
print('port_analysis_1day_df')
pprint(port_analysis_1day_df)

print('report_normal_1day_df')
pprint(report_normal_1day_df)

print('positions_normal_1day_df')
from itertools import islice
print(positions_normal_1day_df)

# 回测绩效分析图
# 回测结果分析图
from qlib.contrib.report import analysis_position
analysis_position.report_graph(report_normal_1day_df)


# 风险分析图
analysis_position.risk_analysis_graph(port_analysis_1day_df, report_normal_1day_df)

#查看全部特征和标签数据
# df_test = dataset.prepare(segments=["test"], data_key="raw")
# 返回（原始数据集中）训练集、验证集、测试集的全部特征和标签数据
df_train, df_valid, df_test = dataset.prepare(segments=["train", "valid", "test"], data_key="raw")
# 不加data_key会返回预处理后的数据
print(df_test)

# 查看标签（即预测对象）的定义
label_expr, label_name = dataset.handler.get_label_config()
print('label_expr', label_expr)
print('label_name', label_name)

# 查看特征定义
fea_expr, fea_name = dataset.handler.get_feature_config()
print('fea_expr', fea_expr)
print()
print('fea_name', fea_name)

# dataset保存为pickle文件
dataset.config(dump_all=True, recursive=True)
dataset.to_pickle(path="dataset.pkl", dump_all=True)