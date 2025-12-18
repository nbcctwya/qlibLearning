import qlib
# region in [REG_CN, REG_US]
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.filter import NameDFilter, ExpressionDFilter
from qlib.data.ops import *

provider_uri = "~/.qlib/qlib_data/cn_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

dd = D.calendar(start_time='2010-01-01', end_time='2019-12-31', freq='day')[-2:]
print(dd)

instruments = D.instruments(market='csi300')
li = D.list_instruments(instruments, start_time='2010-01-01', end_time='2017-12-31', as_list=True)[:6]
print(li)

nameDFilter = NameDFilter(name_rule_re='SH[0-9]{4}55')
instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter])
li = D.list_instruments(instruments=instruments, start_time='2015-01-01', end_time='2016-02-15', as_list=True)
print(li)

expressionDFilter = ExpressionDFilter(rule_expression='$close>Ref($close,1)')
instruments = D.instruments(market='csi300', filter_pipe=[nameDFilter, expressionDFilter])
fields = ['$close', '$volume', 'Ref($close, 1)', 'Mean($close, 3)', '$high-$low']
ddf = D.features(instruments, fields, start_time='2010-01-01', end_time='2017-12-31', freq='day').head().to_string()
print(ddf)

f1 = Feature("high") / Feature("close")
f2 = Feature("open") / Feature("close")
f3 = f1 + f2
f4 = f3 * f3 / f3
data = D.features(["sh600519"], [f4], start_time="20200101")
print(data.head())
