import qlib
from qlib.data import D
from qlib.data.filter import NameDFilter, ExpressionDFilter

# 初始化行情数据
provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri)

#定义股票池
stockpool = D.instruments(market='csi100')
print(stockpool)
# 检索股票池行情价格数据
# data_df = D.features(stockpool, fields=['$close', '$change'])
# factor 为复权因子   OHLC除以复权因子得到复权前的原始价格    volume乘以factor得到复权前的成交量
data_df = D.features(stockpool, fields=['$open', '$high', '$low', '$close', '$change', '$volume'], start_time='2020-7-1', end_time='2020-09-25')
print(data_df)

# 定义和检索派生字段
fields = ['$close', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
data_ddf = D.features(stockpool, fields=fields, start_time='2020-07-01', end_time='2020-09-30')
print(data_ddf)

# 检索股票池股票代码列表
stockpool_list = D.list_instruments(stockpool, as_list=True)
print(stockpool_list)

# 检索交易日历
trade_dates = D.calendar()
print(trade_dates)

# 自行定义股票池
# my_stockpool = ['sh600000', 'sz000001']
my_stockpool = ['sh600000']
data_df = D.features(my_stockpool, fields=['$close', '$change', '$factor'])
print(data_df)

# 自定义股票池(通过instrument目录下的文件来自定义)
stockpool = D.instruments(market='mypool')
data_df = D.features(stockpool, fields=['$close', '$change', '$factor'])
print(data_df)

# 股票代码名字过滤器
nameDFilter = NameDFilter(name_rule_re='SH[0-9]')
# 表达式过滤器
expressionDFilter = ExpressionDFilter(rule_expression='$close >= 5')
# 满足过滤条件的中证100成分股股票池定义
stockpool = D.instruments(market='csi100', filter_pipe=[nameDFilter, expressionDFilter])
data_df = D.features(stockpool, fields=['$close', '$change'])
print(data_df)