# 2022 CFS PUMS 数据内容与模型订单处理

## 1. 数据来源

本项目使用美国交通统计局（BTS）与美国人口普查局联合发布的 **2022 Commodity Flow Survey Public Use Microdata Sample（CFS PUMS）**。本文档依据 2026 年 1 月发布的《2022 CFS Public Use Microdata Sample File – Data Users Guide》编写。

官方 PUMS 是 CSV 文件，共包含 37,576,546 条可用货运微观样本，来自约 50,000 家响应机构。每条记录表示由调查对象定义的一票货物：它具有单一目的地并对应一次运输任务，但可能包含多件货物或使用多辆车。

PUMS 是用于研究的微观样本，不等同于官方公布的 CFS 总量估计。若研究目标是复原总体货运规模，应使用 `WGT_FACTOR` 加权，并优先以官方汇总表校验结果；本项目主要利用它构造具有真实 OD、货量、距离和货类分布的仿真订单。

## 2. 18 个官方字段

| 字段 | 含义 | 本项目用途 |
|---|---|---|
| `SHIPMT_ID` | 随机排序后生成的货运记录编号 | 来源追踪 |
| `ORIG_STATE` | 起点州 FIPS | 辅助地理信息 |
| `ORIG_MA` | 起点都市区编码 | 辅助地理信息 |
| `ORIG_CFS_AREA` | 起点 CFS Area | 城市 1/2 与订单方向 |
| `DEST_STATE` | 终点州 FIPS | 辅助地理信息 |
| `DEST_MA` | 终点都市区编码 | 辅助地理信息 |
| `DEST_CFS_AREA` | 终点 CFS Area | 城市 1/2 与订单方向 |
| `SECTOR` | 发货机构行业部门 | 可用于分层实验 |
| `SCTG` | 两位货物分类编码 | 货类分层、冷链或高价值情景 |
| `MODE` | 运输方式 | 筛选公路货运 |
| `SHIPMT_VALUE` | 货值（美元） | 可选的未服务惩罚校准依据 |
| `SHIPMT_WGHT` | 重量（磅） | 转换为模型 `quantity` |
| `SHIPMT_DIST_GC` | 起终点大圆距离（英里） | 估计线路里程和干线时长 |
| `TEMP_CNTL_YN` | 是否温控 | 订单类别与时限情景 |
| `EXPORT_YN` | 是否出口 | 默认只保留美国国内货运 |
| `EXPORT_CNTRY` | 出口目的地区域 | 默认不使用 |
| `HAZMAT` | 危险品类别 | 订单类别与约束扩展 |
| `WGT_FACTOR` | 该记录代表的总体货运票数 | OD 选择和加权抽样 |

脚本默认保留 `MODE=111`（营运货车）和 `MODE=112`（企业自有货车），不包括 `113`（客户自提）、包裹快递、多式联运和非公路方式。

## 3. 必须明确的数据边界

### 3.1 PUMS 没有订单时限

官方 2022 PUMS 的 18 个字段中没有订单发布时间、要求提货时间、送达时间或截止时间。2017 文件曾包含季度字段，但 2022 PUMS 已移除 `QUARTER`。因此本项目的 `earliest_start` 与 `latest_completion` 是仿真构造字段，不是 CFS 原始观测。

### 3.2 只有大圆距离

2022 CFS 没有计算 routed mileage，公开文件只提供 `SHIPMT_DIST_GC`。脚本采用：

\[
d_l^{route}=\rho\,d_l^{GC}
\]

其中默认绕行系数 \(\rho=1.20\)。该参数应在灵敏度分析中变化，或在确定具体 OD 后用路网服务重新校准。

### 3.3 地理信息经过保密处理

起终点公开到州或 CFS Area，少量记录因保密要求降低了地理或货类精度。默认处理只使用明确的都市区 CFS Area，排除州剩余区域 `99999` 和被抑制区域 `00000`。论文中不能把 CFS Area 描述成企业地址或精确经纬度。

### 3.4 数值经过披露保护

货值、重量和权重可能经过截断、顶码、噪声注入及取整。PUMS 适合构造分布和微观仿真，不应替代官方汇总表中的精确总量。

## 4. 处理流程

处理模块为 `intercity_delivery/data/cfs_processor.py`。

### 4.1 分块读取

官方 CSV 规模很大，脚本通过 `pandas.read_csv(..., chunksize=250000)` 分块读取，只加载转换所需的 12 列。CSV、GZIP 和仅包含一个 CSV 的 ZIP 均可直接读取。

### 4.2 基础筛选

默认条件为：

1. `MODE` 属于 `111,112`；
2. `EXPORT_YN=N`；
3. 起点与终点 CFS Area 不同；
4. 重量和 `WGT_FACTOR` 大于 0；
5. 大圆距离至少 50 英里；
6. 起终点都是明确都市区，而非州剩余或被抑制区域。

### 4.3 选择两城市

推荐在命令行显式指定论文研究的两个 CFS Area。若省略，脚本第一遍扫描所有记录，在两个方向都至少有指定数量记录的 OD 对中，选择“较弱方向的加权货运票数”最大的城市对：

\[
\arg\max_{\{a,b\}}\min\{W_{a\to b},W_{b\to a}\}.
\]

这可以避免选择只有单向强货流的 OD 对。

### 4.4 抽取订单

第二遍扫描只保留选定 OD 对，分别从两个方向抽取近似相等数量的订单。抽样采用以 `WGT_FACTOR` 为权重的无放回蓄水池算法，因此不需要将候选记录全部放入内存，并使总体中更常见的货运类型更可能入选。

定义城市 1 到城市 2 为 `flow="+"`，反方向为 `flow="-"`。

### 4.5 货量转换

模型使用抽象货量单位，默认定义：

\[
1\text{ model unit}=50\text{ lb}.
\]

订单货量计算为：

\[
q_l=\operatorname{clip}\left(
\operatorname{round}\frac{SHIPMT\_WGHT_l}{50},10,300
\right).
\]

这使当前 `capacity_manual=1000` 对应 50,000 磅（约 22.7 公吨）。缩放和截断参数全部可在命令行修改，并记录在元数据中。

### 4.6 干线与最短完成时长

先计算抽样订单估计路线距离的 `WGT_FACTOR` 加权中位数，再计算统一的双城干线时长：

\[
\tau=\left\lceil
\frac{\operatorname{wmedian}(1.2d_l^{GC})}
{50\text{ mph}\times t_0}
\right\rceil.
\]

默认每个城市端至少占用一个服务时段，因此换装路径的最短完成时长为：

\[
p^{min}=\tau+2.
\]

输出元数据中的 `travel_time_periods`、`direct_travel_time_periods`、`T` 和 `t_0_minutes` 是建议同步到 `DeliveryConfig` 的参数。

### 4.7 构造时间窗

由于 PUMS 没有订单日期和截止时间，脚本使用固定随机种子生成：

\[
s_l\sim U\{0,\ldots,T-p^{min}-b_{max}\},
\]

\[
e_l=s_l+p^{min}+b_l,\qquad
b_l\sim U\{b_{min},\ldots,b_{max}\}.
\]

默认 `T=24`、`t_0=60` 分钟、缓冲为 0–5 个时段。若规划期无法容纳完整运输链，脚本会报错，不会静默生成不可行订单。

## 5. 运行方式

项目统一使用 `pavane` 环境：

```powershell
conda activate pavane
```

显式选择两个都市区，例如 Los Angeles–Long Beach 与 San Jose–San Francisco–Oakland：

```powershell
python -m intercity_delivery.data.cfs_processor `
  --input "D:\download\cfs_2022_pums.csv.zip" `
  --output-dir "data\cfs_processed" `
  --city-a 06-348 `
  --city-b 06-488 `
  --num-orders 100 `
  --seed 42
```

自动选择双向货流最充足的都市区对：

```powershell
python -m intercity_delivery.data.cfs_processor `
  --input "D:\download\cfs_2022_pums.csv.zip" `
  --output-dir "data\cfs_processed" `
  --num-orders 100
```

若自动得到的干线时间使 `T=24` 过短，可以增大单时段长度或规划期：

```powershell
python -m intercity_delivery.data.cfs_processor `
  --input "D:\download\cfs_2022_pums.csv.zip" `
  --period-hours 2 `
  --planning-periods 36
```

## 6. 输出文件

| 文件 | 内容 |
|---|---|
| `cfs_model_orders.csv` | 模型字段与每条订单的 CFS 来源字段，便于检查和统计 |
| `cfs_model_orders.json` | `OrderBatch` 兼容字段、完整来源字段和模型参数建议 |
| `cfs_processing_metadata.json` | 输入文件、筛选、抽样、构造参数，以及观测/构造字段清单 |

在 Python 中加载为当前项目的订单三元组：

```python
from intercity_delivery.data.cfs_processor import load_processed_orders
from intercity_delivery.experiments.core import build_delivery_data

orders_tuple = load_processed_orders(
    "data/cfs_processed/cfs_model_orders.json"
)
data = build_delivery_data(config, orders_tuple)
```

载入前应根据 `cfs_processing_metadata.json` 的 `model_recommendations` 同步设置 `DeliveryConfig.T`、`t_0`、`travel_time_periods` 和 `direct_travel_time_periods`。

## 7. 论文中的数据表述

建议将数据方法表述为：

> 本研究从 2022 CFS PUMS 中筛选美国国内公路货运记录，以 CFS Area 构造双城市 OD，使用调查权重进行微观记录抽样，并以观测重量和大圆距离校准订单货量与城际运输尺度。由于公开 PUMS 不提供订单发布时间与交付截止时间，研究在明确的随机种子和缓冲情景下构造时间窗，并对绕行系数、时间窗缓冲和货量缩放进行灵敏度分析。

必须区分：OD、重量、货值、货类、大圆距离和调查权重来自官方数据；路线距离、干线时长、订单出现时段、截止时段、货量缩放和未服务惩罚是模型构造或校准参数。
