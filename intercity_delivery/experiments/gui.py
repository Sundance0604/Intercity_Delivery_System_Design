"""customtkinter 可视化实验工具。

模型、算法、订单三类参数均由 config.py 中的 dataclass 动态生成。界面使用整行
参数标签页和底部双栏输出区，避免参数被压缩在左侧狭窄区域。
"""

import contextlib
import io
import threading
from datetime import datetime
from tkinter import filedialog
from typing import List

import customtkinter as ctk

from intercity_delivery.experiments.core import (
    ExperimentPlan,
    PARAMETER_CONFIGS,
    build_specs,
    get_parameter_groups,
    levels_to_text,
    parse_parameter_levels,
    planned_run_count,
    run_experiment_suite,
)
from intercity_delivery.experiments.solvers import SOLVER_REGISTRY, get_solver_display_name


class QueueWriter(io.TextIOBase):
    """把后台线程输出安全转发到界面日志框。"""

    def __init__(self, callback):
        self.callback = callback

    def write(self, text):
        if text:
            self.callback(text)
        return len(text)

    def flush(self):
        return None


class ExperimentApp(ctk.CTk):
    """城际配送仿真实验主窗口。"""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("城际配送系统仿真实验平台")
        self.geometry("1480x920")
        self.minsize(1180, 760)

        self.plan = ExperimentPlan()
        self.parameter_groups = get_parameter_groups()
        self.sensitivity_parameters = [
            parameter
            for parameters in self.parameter_groups.values()
            for parameter in parameters
        ]
        self.running = False
        self.scenario_vars = {}
        self.solver_vars = {}
        self.data_source_var = ctk.StringVar(value="generated")
        self.real_data_path_var = ctk.StringVar(value="")
        self.fields = {}
        self.sensitivity_fields = {}

        self._build_layout()
        self._refresh_preview()

    def _build_layout(self):
        """构建顶部控制、整行参数区和底部输出区。"""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=3)
        self.grid_rowconfigure(3, weight=2)

        ctk.CTkLabel(
            self,
            text="城际配送系统仿真实验平台",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(14, 8))

        self._build_control_bar()
        self._build_parameter_tabs()
        self._build_output_panel()

    def _build_control_bar(self):
        """在窗口顶部横向放置场景、求解器、运行设置和操作按钮。"""

        bar = ctk.CTkFrame(self)
        bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        bar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        scenario_frame = ctk.CTkFrame(bar, fg_color="transparent")
        scenario_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=10)
        ctk.CTkLabel(
            scenario_frame,
            text="实验场景",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        scenario_labels = {
            "quick": "快速测试",
            "sensitivity": "单因素灵敏度分析",
        }
        for column, (name, label) in enumerate(scenario_labels.items()):
            variable = ctk.BooleanVar(value=(name == "quick"))
            self.scenario_vars[name] = variable
            ctk.CTkCheckBox(
                scenario_frame,
                text=label,
                variable=variable,
                command=self._refresh_preview,
            ).grid(row=1, column=column, sticky="w", padx=(0, 18))

        solver_frame = ctk.CTkFrame(bar, fg_color="transparent")
        solver_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=10)
        ctk.CTkLabel(
            solver_frame,
            text="论文 Solution Approach（可单选或全选）",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        paper_names = ("paper_candidate_mip", "paper_priority_heuristic")
        for column, name in enumerate(paper_names):
            variable = ctk.BooleanVar(value=True)
            self.solver_vars[name] = variable
            ctk.CTkCheckBox(
                solver_frame, text=get_solver_display_name(name), variable=variable,
                command=self._refresh_preview,
            ).grid(row=1, column=column, sticky="w", padx=(0, 18))
        ctk.CTkLabel(solver_frame, text="基准求解器（可选）").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 4)
        )
        baseline_names = [name for name in SOLVER_REGISTRY if name not in paper_names]
        for index, name in enumerate(baseline_names):
            variable = ctk.BooleanVar(value=False)
            self.solver_vars[name] = variable
            ctk.CTkCheckBox(
                solver_frame, text=get_solver_display_name(name), variable=variable,
                command=self._refresh_preview,
            ).grid(row=3 + index // 2, column=index % 2, sticky="w", padx=(0, 18), pady=2)

        settings_frame = ctk.CTkFrame(bar, fg_color="transparent")
        settings_frame.grid(row=0, column=2, sticky="nsew", padx=12, pady=10)
        settings_frame.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(
            settings_frame,
            text="运行设置",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ctk.CTkLabel(settings_frame, text="种子数").grid(
            row=1, column=0, sticky="e", padx=(0, 6)
        )
        self.fields["seed_count"] = self._compact_entry(
            settings_frame, 1, 1, str(self.plan.seed_count)
        )
        ctk.CTkLabel(settings_frame, text="时间限制(秒)").grid(
            row=1, column=2, sticky="e", padx=(14, 6)
        )
        self.fields["time_limit"] = self._compact_entry(
            settings_frame, 1, 3, str(self.plan.time_limit)
        )

        action_frame = ctk.CTkFrame(bar, fg_color="transparent")
        action_frame.grid(row=0, column=3, sticky="e", padx=12, pady=10)
        ctk.CTkButton(
            action_frame,
            text="刷新预览",
            width=110,
            command=self._refresh_preview,
        ).grid(row=0, column=0, padx=5)
        self.run_button = ctk.CTkButton(
            action_frame,
            text="批量运行",
            width=120,
            command=self._run_from_gui,
        )
        self.run_button.grid(row=0, column=1, padx=5)

        data_frame = ctk.CTkFrame(bar, fg_color="transparent")
        data_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkLabel(
            data_frame, text="测试数据", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ctk.CTkRadioButton(
            data_frame, text="生成数据", variable=self.data_source_var,
            value="generated", command=self._refresh_preview,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))
        ctk.CTkRadioButton(
            data_frame, text="真实数据（CFS 处理后 JSON）", variable=self.data_source_var,
            value="real", command=self._refresh_preview,
        ).grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.real_data_entry = ctk.CTkEntry(
            data_frame, textvariable=self.real_data_path_var, width=440,
            placeholder_text="选择 cfs_model_orders.json（仅真实数据模式需要）",
        )
        self.real_data_entry.grid(row=0, column=3, sticky="ew", padx=(0, 8))
        self.real_data_entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
        ctk.CTkButton(
            data_frame, text="浏览", width=70, command=self._browse_real_data
        ).grid(row=0, column=4, sticky="e")
        data_frame.grid_columnconfigure(3, weight=1)

    def _browse_real_data(self):
        path = filedialog.askopenfilename(
            title="选择 CFS 处理后订单 JSON", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")]
        )
        if path:
            self.real_data_path_var.set(path)
            self.data_source_var.set("real")
            self._refresh_preview()

    def _compact_entry(self, parent, row, column, value):
        entry = ctk.CTkEntry(parent, width=90)
        entry.insert(0, value)
        entry.grid(row=row, column=column, sticky="ew")
        entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
        return entry

    def _build_parameter_tabs(self):
        """为三类动态参数建立独立、宽幅标签页。"""

        container = ctk.CTkFrame(self)
        container.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 10))
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            container,
            text="灵敏度参数水平（JSON 数组；每次实验只改变一个参数）",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        tabs = ctk.CTkTabview(container)
        tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        for source, (category_label, _config_type) in PARAMETER_CONFIGS.items():
            tab = tabs.add(category_label)
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            scroll = ctk.CTkScrollableFrame(tab)
            scroll.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            scroll.grid_columnconfigure((1, 3), weight=1)
            self._populate_parameter_group(
                scroll, self.parameter_groups[source]
            )

    def _populate_parameter_group(self, frame, parameters):
        """把一类参数分成左右两组，充分利用横向空间。"""

        split_index = (len(parameters) + 1) // 2
        for index, parameter in enumerate(parameters):
            group = 0 if index < split_index else 1
            local_row = index if group == 0 else index - split_index
            label_column = group * 2
            entry_column = label_column + 1
            base_text = levels_to_text([parameter.base_value])
            ctk.CTkLabel(
                frame,
                text=f"{parameter.field_name}\n基准值 {base_text}",
                anchor="w",
                justify="left",
            ).grid(
                row=local_row,
                column=label_column,
                sticky="w",
                padx=(10, 8),
                pady=7,
            )
            entry = ctk.CTkEntry(frame)
            entry.insert(
                0,
                levels_to_text(self.plan.sensitivity_levels[parameter.key]),
            )
            entry.grid(
                row=local_row,
                column=entry_column,
                sticky="ew",
                padx=(0, 18),
                pady=7,
            )
            entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
            self.sensitivity_fields[parameter.key] = entry

    def _build_output_panel(self):
        """底部横向展示实验预览和运行日志。"""

        panel = ctk.CTkFrame(self)
        panel.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 18))
        panel.grid_columnconfigure((0, 1), weight=1)
        panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            panel,
            text="实验计划预览",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            panel,
            text="运行日志",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=1, sticky="w", padx=12, pady=(10, 4))

        self.preview = ctk.CTkTextbox(panel)
        self.preview.grid(
            row=1, column=0, sticky="nsew", padx=(12, 6), pady=(0, 12)
        )
        self.log = ctk.CTkTextbox(panel)
        self.log.grid(
            row=1, column=1, sticky="nsew", padx=(6, 12), pady=(0, 12)
        )

    def _selected_scenarios(self) -> List[str]:
        return [
            name for name, variable in self.scenario_vars.items() if variable.get()
        ]

    def _selected_solvers(self) -> List[str]:
        return [
            name for name, variable in self.solver_vars.items() if variable.get()
        ]

    def _read_plan(self) -> ExperimentPlan:
        """读取固定运行设置和三类动态灵敏度水平。"""

        sensitivity_levels = {}
        for parameter in self.sensitivity_parameters:
            sensitivity_levels[parameter.key] = parse_parameter_levels(
                self.sensitivity_fields[parameter.key].get(),
                parameter.base_value,
            )
        return ExperimentPlan(
            seed_count=int(self.fields["seed_count"].get()),
            time_limit=int(self.fields["time_limit"].get()),
            sensitivity_levels=sensitivity_levels,
        )

    def _refresh_preview(self):
        """解析当前输入并刷新计划，不调用求解器。"""

        try:
            plan = self._read_plan()
            solvers = self._selected_solvers()
            specs = build_specs(self._selected_scenarios(), plan)
            run_count = planned_run_count(specs, solvers)
        except Exception as exc:
            self.preview.delete("1.0", "end")
            self.preview.insert("end", f"参数暂不可解析：{exc}")
            return

        category_counts = {"model": 0, "algorithm": 0, "order": 0}
        for spec in specs:
            if spec.sensitivity_parameter:
                category_counts[spec.sensitivity_parameter.split(".", 1)[0]] += 1

        lines = [
            f"算例规格数：{len(specs)}",
            f"数据来源：{'真实 CFS 数据' if self.data_source_var.get() == 'real' else '程序生成数据'}",
            f"数据文件：{self.real_data_path_var.get() or '未选择'}" if self.data_source_var.get() == "real" else "",
            f"实际求解次数：{run_count}",
            (
                "分类：模型 {model} / 算法 {algorithm} / 订单 {order}".format(
                    **category_counts
                )
            ),
            "说明：精确 MIP 自动跳过算法参数灵敏度规格。",
            "",
            "前 30 个算例：",
        ]
        for spec in specs[:30]:
            sensitivity = (
                f"{spec.sensitivity_parameter}={spec.sensitivity_value}"
                if spec.sensitivity_parameter
                else "快速连通性检查"
            )
            lines.append(
                f"{spec.experiment_id} | {spec.scenario} | "
                f"orders={spec.order_config.num_orders} | seed={spec.seed} | "
                f"{sensitivity}"
            )
        if len(specs) > 30:
            lines.append(f"... 另有 {len(specs) - 30} 个算例未显示")

        self.preview.delete("1.0", "end")
        self.preview.insert("end", "\n".join(lines))

    def _append_log(self, text: str):
        self.after(0, self._append_log_on_ui_thread, text)

    def _append_log_on_ui_thread(self, text: str):
        self.log.insert("end", text)
        self.log.see("end")

    def _run_from_gui(self):
        """校验参数并启动后台实验线程。"""

        if self.running:
            self._append_log("\n当前已有实验正在运行，请等待完成。\n")
            return

        try:
            plan = self._read_plan()
            specs = build_specs(self._selected_scenarios(), plan)
            solver_names = self._selected_solvers()
            if not solver_names:
                raise ValueError("请至少选择一个求解方式。")
            if self.data_source_var.get() == "real" and not self.real_data_path_var.get().strip():
                raise ValueError("真实数据模式必须选择 cfs_model_orders.json。")
            if planned_run_count(specs, solver_names) <= 0:
                raise ValueError("当前参数类别与所选求解器之间没有可执行组合。")
        except Exception as exc:
            self._append_log(f"\n[参数错误] {exc}\n")
            return

        self.running = True
        self.run_button.configure(state="disabled")
        self.log.delete("1.0", "end")
        self._append_log(
            f"准备运行 {len(specs)} 个规格，"
            f"{planned_run_count(specs, solver_names)} 次求解。\n"
        )
        threading.Thread(
            target=self._run_worker,
            args=(specs, solver_names, self.data_source_var.get(), self.real_data_path_var.get().strip()),
            daemon=True,
        ).start()

    def _run_worker(self, specs, solver_names, data_source, real_data_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = QueueWriter(self._append_log)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                run_experiment_suite(
                    specs, solver_names, timestamp,
                    data_source=data_source,
                    real_data_path=real_data_path or None,
                )
        except Exception as exc:
            self._append_log(f"\n[错误] {exc}\n")
        finally:
            self.running = False
            self.after(0, lambda: self.run_button.configure(state="normal"))
            self._append_log("\n运行线程已结束。\n")


def launch_gui():
    ExperimentApp().mainloop()
