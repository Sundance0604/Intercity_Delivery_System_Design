"""customtkinter 可视化实验工具。

本文件只负责界面展示和用户交互，不直接写优化模型，也不直接生成订单。
界面读取用户输入后，会调用 experiment_core.py 中的函数生成实验计划并运行。
"""

import contextlib
import io
import threading
from dataclasses import asdict
from datetime import datetime
from typing import List

import customtkinter as ctk

from experiment_core import (
    ExperimentPlan,
    build_specs,
    buffer_ranges_to_text,
    float_list_to_text,
    int_list_to_text,
    parse_buffer_ranges,
    parse_float_list,
    parse_int_list,
    run_experiment_suite,
)
from solvers import SOLVER_REGISTRY, get_solver_display_name


class QueueWriter(io.TextIOBase):
    """把后台线程中的 print 输出安全地转发到界面日志框。"""

    def __init__(self, callback):
        self.callback = callback

    def write(self, text):
        if text:
            self.callback(text)
        return len(text)

    def flush(self):
        return None


class ExperimentApp(ctk.CTk):
    """仿真实验主窗口。

    这个类的职责是：
    1. 展示用户可修改的实验参数；
    2. 预览即将生成的算例；
    3. 选择求解器；
    4. 在后台线程中批量运行实验，避免界面卡死。
    """

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("城际配送系统仿真实验平台")
        self.geometry("1180x780")
        self.minsize(1050, 700)

        self.plan = ExperimentPlan()
        self.running = False
        self.scenario_vars = {}
        self.solver_vars = {}
        self.fields = {}

        self._build_layout()
        self._refresh_preview()

    def _build_layout(self):
        """搭建整体布局。"""

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            self,
            text="城际配送系统仿真实验平台",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        left_panel = ctk.CTkFrame(self, width=360)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(18, 8), pady=(0, 18))
        left_panel.grid_columnconfigure(0, weight=1)

        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=(0, 18))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        self._build_selector_panel(left_panel)
        self._build_parameter_panel(left_panel)
        self._build_action_panel(left_panel)
        self._build_preview_and_log(right_panel)

    def _build_selector_panel(self, parent):
        """场景和求解器选择区。"""

        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="实验场景", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 4)
        )

        scenario_labels = {
            "quick": "快速测试 quick",
            "baseline": "小规模基准 baseline",
            "scale": "规模扩展 scale",
            "sensitivity": "灵敏度分析 sensitivity",
        }
        for row, (name, label) in enumerate(scenario_labels.items(), start=1):
            var = ctk.BooleanVar(value=(name == "quick"))
            self.scenario_vars[name] = var
            ctk.CTkCheckBox(frame, text=label, variable=var, command=self._refresh_preview).grid(
                row=row, column=0, sticky="w", padx=12, pady=4
            )

        solver_start = len(scenario_labels) + 1
        ctk.CTkLabel(frame, text="求解方式", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=solver_start, column=0, sticky="w", padx=12, pady=(14, 4)
        )

        for row, name in enumerate(SOLVER_REGISTRY.keys(), start=solver_start + 1):
            var = ctk.BooleanVar(value=(name == "exact_mip"))
            self.solver_vars[name] = var
            ctk.CTkCheckBox(
                frame,
                text=get_solver_display_name(name),
                variable=var,
                command=self._refresh_preview,
            ).grid(row=row, column=0, sticky="w", padx=12, pady=4)

    def _build_parameter_panel(self, parent):
        """参数输入区。所有输入框都写中文标签，便于以后手工修改。"""

        frame = ctk.CTkScrollableFrame(parent, label_text="参数范围")
        frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        parent.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        rows = [
            ("seed_count", "每个水平随机种子数", str(self.plan.seed_count)),
            ("time_limit", "单算例时间限制(秒)", str(self.plan.time_limit)),
            ("quick_orders", "quick 订单数", str(self.plan.quick_orders)),
            ("baseline_order_sizes", "baseline 订单规模", int_list_to_text(self.plan.baseline_order_sizes)),
            ("scale_order_sizes", "scale 订单规模", int_list_to_text(self.plan.scale_order_sizes)),
            ("scale_auto_fleet", "scale 自动车数/城市", str(self.plan.scale_auto_fleet)),
            ("scale_manual_fleet", "scale 人工车数/城市", str(self.plan.scale_manual_fleet)),
            ("sensitivity_orders", "灵敏度订单数", str(self.plan.sensitivity_orders)),
            ("sensitivity_base_auto", "灵敏度基准自动车", str(self.plan.sensitivity_base_auto)),
            ("sensitivity_base_manual", "灵敏度基准人工车", str(self.plan.sensitivity_base_manual)),
            ("auto_fleet_levels", "自动车数量水平", int_list_to_text(self.plan.auto_fleet_levels)),
            ("auto_cost_levels", "自动车成本水平", float_list_to_text(self.plan.auto_cost_levels)),
            ("manual_fleet_levels", "人工车数量水平", int_list_to_text(self.plan.manual_fleet_levels)),
            ("time_window_buffers", "时间窗缓冲水平", buffer_ranges_to_text(self.plan.time_window_buffers)),
            ("large_order_probs", "大订单比例水平", float_list_to_text(self.plan.large_order_probs)),
        ]

        for row, (key, label, value) in enumerate(rows):
            ctk.CTkLabel(frame, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=5)
            entry = ctk.CTkEntry(frame)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
            self.fields[key] = entry

    def _build_action_panel(self, parent):
        """按钮区。"""

        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(frame, text="刷新预览", command=self._refresh_preview).grid(
            row=0, column=0, sticky="ew", padx=8, pady=10
        )
        ctk.CTkButton(frame, text="批量运行", command=self._run_from_gui).grid(
            row=0, column=1, sticky="ew", padx=8, pady=10
        )

    def _build_preview_and_log(self, parent):
        """右侧预览和日志区。"""

        preview_label = ctk.CTkLabel(parent, text="实验计划预览", font=ctk.CTkFont(size=16, weight="bold"))
        preview_label.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

        self.preview = ctk.CTkTextbox(parent, height=260)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))

        log_label = ctk.CTkLabel(parent, text="运行日志", font=ctk.CTkFont(size=16, weight="bold"))
        log_label.grid(row=2, column=0, sticky="w", padx=12, pady=(4, 4))

        self.log = ctk.CTkTextbox(parent, height=260)
        self.log.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        parent.grid_rowconfigure(3, weight=1)

    def _selected_scenarios(self) -> List[str]:
        return [name for name, var in self.scenario_vars.items() if var.get()]

    def _selected_solvers(self) -> List[str]:
        return [name for name, var in self.solver_vars.items() if var.get()]

    def _read_plan(self) -> ExperimentPlan:
        """从界面输入框读取参数并转换为 ExperimentPlan。"""

        return ExperimentPlan(
            seed_count=int(self.fields["seed_count"].get()),
            time_limit=int(self.fields["time_limit"].get()),
            quick_orders=int(self.fields["quick_orders"].get()),
            baseline_order_sizes=parse_int_list(self.fields["baseline_order_sizes"].get()),
            scale_order_sizes=parse_int_list(self.fields["scale_order_sizes"].get()),
            scale_auto_fleet=int(self.fields["scale_auto_fleet"].get()),
            scale_manual_fleet=int(self.fields["scale_manual_fleet"].get()),
            sensitivity_orders=int(self.fields["sensitivity_orders"].get()),
            sensitivity_base_auto=int(self.fields["sensitivity_base_auto"].get()),
            sensitivity_base_manual=int(self.fields["sensitivity_base_manual"].get()),
            auto_fleet_levels=parse_int_list(self.fields["auto_fleet_levels"].get()),
            auto_cost_levels=parse_float_list(self.fields["auto_cost_levels"].get()),
            manual_fleet_levels=parse_int_list(self.fields["manual_fleet_levels"].get()),
            time_window_buffers=parse_buffer_ranges(self.fields["time_window_buffers"].get()),
            large_order_probs=parse_float_list(self.fields["large_order_probs"].get()),
        )

    def _refresh_preview(self):
        """刷新实验计划预览。

        这个函数不会调用 Gurobi，只是根据参数计算将生成多少个算例。
        """

        try:
            plan = self._read_plan()
            scenarios = self._selected_scenarios()
            solvers = self._selected_solvers()
            specs = build_specs(scenarios, plan)
        except Exception as exc:
            self.preview.delete("1.0", "end")
            self.preview.insert("end", f"参数暂不可解析：{exc}")
            return

        lines = [
            f"场景数量：{len(specs)}",
            f"求解器数量：{len(solvers)}",
            f"预计结果行数：{len(specs) * len(solvers)}",
            "",
            "前 30 个算例：",
        ]
        for spec in specs[:30]:
            lines.append(
                f"{spec.experiment_id} | {spec.scenario} | orders={spec.num_orders} | "
                f"seed={spec.seed} | N_auto={spec.config.N_auto[1]} | "
                f"N_manual={spec.config.N_manual[1]} | cost_auto={spec.config.cost_auto} | "
                f"buffer={spec.buffer_range} | large_prob={spec.large_order_prob}"
            )
        if len(specs) > 30:
            lines.append(f"... 另有 {len(specs) - 30} 个算例未显示")

        self.preview.delete("1.0", "end")
        self.preview.insert("end", "\n".join(lines))

    def _append_log(self, text: str):
        """线程安全地追加日志。"""

        self.after(0, self._append_log_on_ui_thread, text)

    def _append_log_on_ui_thread(self, text: str):
        self.log.insert("end", text)
        self.log.see("end")

    def _run_from_gui(self):
        """读取界面参数并启动后台实验线程。"""

        if self.running:
            self._append_log("\n当前已有实验正在运行，请等待完成。\n")
            return

        try:
            plan = self._read_plan()
            specs = build_specs(self._selected_scenarios(), plan)
            solver_names = self._selected_solvers()
            if not solver_names:
                raise ValueError("请至少选择一个求解方式。")
        except Exception as exc:
            self._append_log(f"\n[参数错误] {exc}\n")
            return

        self.running = True
        self.log.delete("1.0", "end")
        self._append_log(f"准备运行 {len(specs)} 个算例，{len(solver_names)} 个求解器。\n")

        worker = threading.Thread(target=self._run_worker, args=(specs, solver_names), daemon=True)
        worker.start()

    def _run_worker(self, specs, solver_names):
        """后台线程实际执行批量实验。"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = QueueWriter(self._append_log)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                run_experiment_suite(specs, solver_names, timestamp)
        except Exception as exc:
            self._append_log(f"\n[错误] {exc}\n")
        finally:
            self.running = False
            self._append_log("\n运行线程已结束。\n")


def launch_gui():
    """启动图形界面。"""

    app = ExperimentApp()
    app.mainloop()
