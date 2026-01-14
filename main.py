import pandas as pd
import numpy as np
import gurobipy as gp
import json
import random
import os
import time
from datetime import datetime
from dataclasses import replace, asdict
from itertools import product

# 引入你的模块
from config import DeliveryConfig
from data_loader import DataLoader, DeliveryData, OrderBatch
from optimizer import Optimizer

# ==========================================
# 1. 随机订单生成器 (增加规模参数)
# ==========================================
def generate_random_orders(config: DeliveryConfig, num_orders=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    pos_orders = {}
    neg_orders = {}
    all_orders = {}
    
    for l in range(1, num_orders + 1):
        flow = "+" if random.random() > 0.5 else "-"
        # 增加随机性：长短途混合
        min_duration = config.travel_time_periods + 1
        max_start = config.T - min_duration - 1
        
        if max_start <= 0: # 防止T设置过小导致报错
            earliest_start = 0
            latest_completion = config.T
        else:
            earliest_start = random.randint(0, max_start)
            # 结束时间 = 开始 + 行驶时间 + 随机缓冲(0-5个时间段)
            buffer = random.randint(0, 5)
            latest_completion = min(config.T, earliest_start + min_duration + buffer)
        
        # 需求量波动：小包裹(10-50) vs 大宗货物(100-300)
        quantity = random.randint(10, 50) if random.random() > 0.3 else random.randint(100, 300)
        
        order = OrderBatch(
            batch_id=l,
            flow=flow,
            quantity=quantity,
            earliest_start=earliest_start,
            latest_completion=latest_completion,
            penalty_lost=config.penalty_lost * quantity
        )
        
        all_orders[l] = order
        if flow == "+":
            pos_orders[l] = order
        else:
            neg_orders[l] = order
            
    return pos_orders, neg_orders, all_orders

# ==========================================
# 2. 单次实验运行器 (增强版：增加详细日志)
# ==========================================
def run_single_experiment(experiment_id, config, orders_tuple):
    start_time = time.time()
    pos, neg, all_ord = orders_tuple
    
    # 1. 数据加载
    loader = DataLoader(config)
    # 注意：这里调用加上 list() 转换后的 generate_arcs_manual
    m1, m2 = loader.generate_arcs_manual() 
    auto = loader.generate_arcs_auto()
    sets_m1, sets_m2, sets_auto = loader.generate_sets(m1, m2, auto)
    epsilon = loader.generate_epsilon_sets(pos, neg, m1, m2)
    coeff1, coeff2 = loader.pre_inverse_count(m1, m2)
    
    data = DeliveryData(
        arcs_manual_1=m1, arcs_manual_2=m2, arcs_auto=auto,
        sets_manual_1=sets_m1, sets_manual_2=sets_m2, sets_auto=sets_auto,
        cap_coeff_1=coeff1, cap_coeff_2=coeff2,
        pos_orders=pos, neg_orders=neg, all_orders=all_ord,
        epsilon_sets=epsilon
    )
    
    # 2. 求解
    opt = Optimizer(config, data)
    opt.setup_variables()
    opt.set_objective()
    opt.set_constraints()
    
    # 设置求解时间限制 (防止大规模卡死)
    opt.model.setParam('TimeLimit', 500) # 5分钟限制
    opt.model.setParam('OutputFlag', 0)
    opt.model.optimize()
    
    solve_time = time.time() - start_time
    
    # 3. 结果提取
    result_summary = {
        "Exp_ID": experiment_id,
        "Status": opt.model.Status,
        "Solve_Time_Sec": round(solve_time, 2),
        "Num_Orders": len(all_ord),
        # 记录关键参数
        "Param_N_Auto": config.N_auto[1],
        "Param_N_Manual": config.N_manual[1],
        "Param_Cost_Auto": config.cost_auto,
        # 结果指标
        "Total_Cost": None,
        "Unserved_Rate": None,
        "Auto_Usage": 0,
        "Manual_Usage": 0
    }
    
    detailed_log = None # 用于JSON保存
    
    # 修改这里的逻辑：检查是否找到了解 
    if opt.model.SolCount > 0: 
        # 1. 记录结果 (这些逻辑不变)
        result_summary["Total_Cost"] = opt.model.ObjVal
        total_demand = sum(o.quantity for o in all_ord.values())
        unserved_amount = sum(v.X for v in opt.z_unserved.values())
        result_summary["Unserved_Rate"] = round(unserved_amount / total_demand, 4) if total_demand > 0 else 0
        result_summary["Auto_Usage"] = sum(v.X for v in opt.y_auto.values())
        result_summary["Manual_Usage"] = sum(v.X for v in opt.x_manual.values())
        
        # --- 【新增部分】状态检查与提示 ---
        if opt.model.Status == gp.GRB.OPTIMAL:
            print(f"  [成功] 找到全局最优解！Cost = {result_summary['Total_Cost']:.2f}")
            
        elif opt.model.Status == gp.GRB.TIME_LIMIT:
            # 计算一下当前的 Gap (离最优下界的差距)
            gap = opt.model.MIPGap * 100
            print(f"  [警告] 达到时间限制 ({opt.model.Params.TimeLimit}s)！")
            print(f"  [提示] 当前解可能不是最优解 (MIP Gap: {gap:.2f}%)")
            print(f"        当前找到的最好 Cost = {result_summary['Total_Cost']:.2f}")
            
        # 生成详细日志 
        detailed_log = {
            "config": asdict(config), # 需要把 config 变成字典
            "orders": {k: asdict(v) for k,v in all_ord.items()},
            "solution": {
                "y_auto": {str(k): v.X for k, v in opt.y_auto.items() if v.X > 0.1},
                # z_unserved 记录哪些订单没送完
                "z_unserved": {k: v.X for k, v in opt.z_unserved.items() if v.X > 0.1}
            }
        }
        print(f"Exp {experiment_id} | Orders={len(all_ord)} | Auto={config.N_auto[1]} | Cost={result_summary['Total_Cost']}")
    else:
        # 如果 SolCount == 0，说明连一个可行解都没找到
        print(f"  [失败] 未找到任何可行解。Gurobi 状态码: {opt.model.Status}")

    
    return result_summary, detailed_log

# ==========================================
# 3. 实验场景运行器
# ==========================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_summaries = []
    experiment_type = input("选择仿真类型：快速测试(1)或复杂实验(2)")
    if experiment_type.strip() == "1":
        print("\n=== 开始场景 A: 多因素参数敏感性分析 ===")
        
        # 1. 定义因子水平
        levels_n_auto = [10, 20, 30]          # 自动车数量
        levels_cost_auto = [10.0, 15.0, 20.0] # 自动车成本 (原价15, 测试降价和涨价)
        levels_n_manual = [20, 40]            # 人工车数量
        
        # 生成基础订单 (中等规模)
        base_cfg = DeliveryConfig()
        fixed_orders = generate_random_orders(base_cfg, num_orders=50, seed=100)
        
        # 2. 生成所有组合 (3*3*2 = 18组实验)
        param_combinations = list(product(levels_n_auto, levels_cost_auto, levels_n_manual))
        
        for i, (n_auto, c_auto, n_manual) in enumerate(param_combinations):
            # 创建特定配置
            exp_cfg = replace(base_cfg, 
                            N_auto={1: n_auto, 2: n_auto},
                            cost_auto=c_auto,
                            N_manual={1: n_manual, 2: n_manual})
            
            # 运行
            res, _ = run_single_experiment(f"A_{i+1}", exp_cfg, fixed_orders)
            all_summaries.append(res)
    if experiment_type.strip() == "2":
    
        print("\n=== 开始场景 B: 大规模订单压力测试 ===")
        
        scale_levels = [100, 200, 500] # 测试 100 到 500 个订单
        base_cfg_scale = DeliveryConfig(
            N_auto={1: 50, 2: 50},       # 增加车辆以应对大规模订单
            N_manual={1: 100, 2: 100}
        )
        
        for i, n_orders in enumerate(scale_levels):
            # 每次生成不同规模的新订单集
            scale_orders = generate_random_orders(base_cfg_scale, num_orders=n_orders, seed=200+i)
            
            res, details = run_single_experiment(f"B_{n_orders}", base_cfg_scale, scale_orders)
            all_summaries.append(res)
            
            # 保存大规模实验的详细日志到 JSON (选做，防止文件过大)
            if details:
                with open(f"results/detail_exp_B_{n_orders}_{timestamp}.json", "w") as f:
                    json.dump(details, f, indent=4)

    # --- 保存最终汇总表 ---
    df = pd.DataFrame(all_summaries)
    csv_filename = f"results/full_experiment_summary_{timestamp}.csv"
    
    # 调整列顺序，好看一点
    cols = ["Exp_ID", "Num_Orders", "Param_N_Auto", "Param_Cost_Auto", "Param_N_Manual", 
            "Total_Cost", "Unserved_Rate", "Solve_Time_Sec", "Status"]
    df = df[cols]
    
    df.to_csv(csv_filename, index=False)
    print(f"\n所有测试完成！汇总结果已保存至: {csv_filename}")
    print(df)