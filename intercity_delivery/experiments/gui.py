"""customtkinter 可视化实验工具。

模型、算法、订单三类参数均由 config.py 中的 dataclass 动态生成。界面使用整行
参数标签页和底部双栏输出区，避免参数被压缩在左侧狭窄区域。
"""

import contextlib
import io
import threading
from datetime import datetime
from tkinter import filedialog
from typing import List, Optional, Tuple

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
from intercity_delivery.data.cfs_catalog import (
    CFSSQLiteCatalog,
    cfs_area_name,
    inspect_cfs_sqlite,
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
        self.sqlite_columns_var = ctk.StringVar(value="尚未加载 SQLite")
        self.city_a_var = ctk.StringVar(value="")
        self.city_b_var = ctk.StringVar(value="")
        self.city_pair_stats_var = ctk.StringVar(value="尚未选择城市对")
        self.city_label_to_code = {}
        self.city_pair_records = {}
        self.loaded_sqlite_path = ""
        self.catalog_loading = False
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
        data_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            data_frame, text="测试数据", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ctk.CTkRadioButton(
            data_frame,
            text="生成数据",
            variable=self.data_source_var,
            value="generated",
            command=self._refresh_preview,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))
        ctk.CTkRadioButton(
            data_frame,
            text="真实数据（CFS SQLite）",
            variable=self.data_source_var,
            value="real",
            command=self._refresh_preview,
        ).grid(row=0, column=2, sticky="w")

        ctk.CTkLabel(data_frame, text="SQLite 文件").grid(
            row=1, column=0, sticky="e", padx=(0, 8), pady=(7, 2)
        )
        self.real_data_entry = ctk.CTkEntry(
            data_frame,
            textvariable=self.real_data_path_var,
            placeholder_text="选择 cfs_2022_pums.sqlite",
        )
        self.real_data_entry.grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(7, 2)
        )
        self.real_data_entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
        ctk.CTkButton(
            data_frame, text="浏览", width=70, command=self._browse_real_data
        ).grid(row=1, column=3, sticky="e", padx=(0, 6), pady=(7, 2))
        self.load_sqlite_button = ctk.CTkButton(
            data_frame, text="加载", width=70, command=self._start_load_sqlite
        )
        self.load_sqlite_button.grid(row=1, column=4, sticky="e", pady=(7, 2))

        ctk.CTkLabel(data_frame, text="shipments 列名").grid(
            row=2, column=0, sticky="ne", padx=(0, 8), pady=4
        )
        ctk.CTkLabel(
            data_frame,
            textvariable=self.sqlite_columns_var,
            anchor="w",
            justify="left",
            wraplength=1080,
        ).grid(row=2, column=1, columnspan=4, sticky="ew", pady=4)

        ctk.CTkLabel(data_frame, text="城市 1").grid(
            row=3, column=0, sticky="e", padx=(0, 8), pady=(2, 0)
        )
        self.city_a_combo = ctk.CTkComboBox(
            data_frame,
            variable=self.city_a_var,
            values=["请先加载 SQLite"],
            state="readonly",
            command=lambda _value: self._on_city_a_selected(),
        )
        self.city_a_combo.grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(2, 0)
        )
        ctk.CTkLabel(data_frame, text="城市 2").grid(
            row=3, column=3, sticky="e", padx=(0, 8), pady=(2, 0)
        )
        self.city_b_combo = ctk.CTkComboBox(
            data_frame,
            variable=self.city_b_var,
            values=["请先加载 SQLite"],
            state="readonly",
            command=lambda _value: self._on_city_pair_selected(),
        )
        self.city_b_combo.grid(row=3, column=4, sticky="ew", pady=(2, 0))
        ctk.CTkLabel(
            data_frame,
            textvariable=self.city_pair_stats_var,
            anchor="w",
        ).grid(row=4, column=1, columnspan=4, sticky="w", pady=(3, 0))
        self.city_a_combo.set("请先加载 SQLite")
        self.city_b_combo.set("请先加载 SQLite")
    def _browse_real_data(self):
        path = filedialog.askopenfilename(
            title="选择 CFS SQLite",
            filetypes=[
                ("SQLite", "*.sqlite *.sqlite3 *.db"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.real_data_path_var.set(path)
            self.data_source_var.set("real")
            self._start_load_sqlite()

    def _start_load_sqlite(self):
        path = self.real_data_path_var.get().strip()
        if not path:
            self._append_log("\n[数据错误] 请先选择 SQLite 文件。\n")
            return
        if self.catalog_loading:
            return
        self.catalog_loading = True
        self.load_sqlite_button.configure(state="disabled", text="加载中")
        self.sqlite_columns_var.set("正在读取 shipments 表结构和双向城市对……")
        self.city_label_to_code = {}
        self.city_pair_records = {}
        self.city_a_var.set("")
        self.city_b_var.set("")
        self.city_pair_stats_var.set("正在加载城市对……")
        self._append_log(f"\n正在加载 CFS SQLite：{path}\n")
        threading.Thread(
            target=self._load_sqlite_worker,
            args=(path,),
            daemon=True,
        ).start()

    def _load_sqlite_worker(self, path: str):
        try:
            catalog = inspect_cfs_sqlite(path)
        except Exception as exc:
            self.after(0, self._on_sqlite_load_error, str(exc))
            return
        self.after(0, self._apply_sqlite_catalog, catalog)

    def _apply_sqlite_catalog(self, catalog: CFSSQLiteCatalog):
        self.catalog_loading = False
        self.loaded_sqlite_path = catalog.database_path
        self.real_data_path_var.set(catalog.database_path)
        self.load_sqlite_button.configure(state="normal", text="重新加载")
        columns_text = "，".join(
            f"{name} ({sql_type or '未声明类型'})"
            for name, sql_type in catalog.columns
        )
        self.sqlite_columns_var.set(
            f"{len(catalog.columns)} 列：{columns_text}"
        )
        self.city_pair_records = {
            (pair.city_a, pair.city_b): pair
            for pair in catalog.city_pairs
        }
        city_codes = sorted(
            {
                code
                for pair in catalog.city_pairs
                for code in (pair.city_a, pair.city_b)
            }
        )
        labels = {
            code: f"{cfs_area_name(code)} [{code}]"
            for code in city_codes
        }
        self.city_label_to_code = {
            label: code for code, label in labels.items()
        }
        city_a_values = sorted(labels.values())
        if not city_a_values:
            self.city_a_combo.configure(values=["没有可选城市"])
            self.city_b_combo.configure(values=["没有可选城市"])
            self.city_a_combo.set("没有可选城市")
            self.city_b_combo.set("没有可选城市")
            self.city_pair_stats_var.set("SQLite 中没有双向都市区城市对")
        else:
            self.city_a_combo.configure(values=city_a_values)
            self.city_a_combo.set(city_a_values[0])
            self.city_a_var.set(city_a_values[0])
            self._on_city_a_selected()
        self._append_log(
            f"SQLite 加载完成：{len(catalog.columns)} 列，"
            f"{len(catalog.city_pairs)} 个双向都市区城市对。\n"
        )
        self._refresh_preview()

    def _on_sqlite_load_error(self, message: str):
        self.catalog_loading = False
        self.loaded_sqlite_path = ""
        self.load_sqlite_button.configure(state="normal", text="加载")
        self.sqlite_columns_var.set("SQLite 加载失败")
        self._append_log(f"[SQLite 错误] {message}\n")
        self._refresh_preview()

    def _on_city_a_selected(self):
        city_a = self.city_label_to_code.get(self.city_a_var.get())
        if city_a is None:
            self.city_pair_stats_var.set("请选择城市 1")
            self._refresh_preview()
            return
        partner_codes = sorted(
            city_b if city_a == pair_a else pair_a
            for pair_a, city_b in self.city_pair_records
            if city_a in {pair_a, city_b}
        )
        code_to_label = {
            code: label for label, code in self.city_label_to_code.items()
        }
        values = sorted(code_to_label[code] for code in partner_codes)
        self.city_b_combo.configure(values=values or ["没有双向城市"])
        selected = values[0] if values else "没有双向城市"
        self.city_b_combo.set(selected)
        self.city_b_var.set(selected)
        self._on_city_pair_selected()

    def _on_city_pair_selected(self):
        pair = self._selected_city_pair()
        if pair is None:
            self.city_pair_stats_var.set("请选择有效的双向城市对")
        else:
            record = self.city_pair_records[tuple(sorted(pair))]
            if pair[0] == record.city_a:
                forward, reverse = record.records_a_to_b, record.records_b_to_a
            else:
                forward, reverse = record.records_b_to_a, record.records_a_to_b
            self.city_pair_stats_var.set(
                f"原始记录数：城市 1→城市 2 {forward:,}，"
                f"城市 2→城市 1 {reverse:,}"
            )
        self._refresh_preview()

    def _selected_city_pair(self) -> Optional[Tuple[str, str]]:
        city_a = self.city_label_to_code.get(self.city_a_var.get())
        city_b = self.city_label_to_code.get(self.city_b_var.get())
        if not city_a or not city_b:
            return None
        key = tuple(sorted((city_a, city_b)))
        return (city_a, city_b) if key in self.city_pair_records else None

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
            f"数据来源：{'真实 CFS SQLite' if self.data_source_var.get() == 'real' else '程序生成数据'}",
            f"SQLite：{self.real_data_path_var.get() or '未选择'}" if self.data_source_var.get() == "real" else "",
            (
                f"城市对：{self.city_a_var.get() or '未选择'} → {self.city_b_var.get() or '未选择'}"
                if self.data_source_var.get() == "real"
                else ""
            ),
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
            real_city_pair = None
            if self.data_source_var.get() == "real":
                path = self.real_data_path_var.get().strip()
                if not path:
                    raise ValueError("真实数据模式必须选择 CFS SQLite。")
                if self.catalog_loading:
                    raise ValueError("SQLite 仍在加载，请稍候。")
                if self.loaded_sqlite_path != path:
                    raise ValueError("SQLite 路径已改变，请点击“加载”。")
                real_city_pair = self._selected_city_pair()
                if real_city_pair is None:
                    raise ValueError("请选择一个双向城市对。")
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
            args=(
                specs,
                solver_names,
                self.data_source_var.get(),
                self.real_data_path_var.get().strip(),
                real_city_pair,
            ),
            daemon=True,
        ).start()

    def _run_worker(
        self,
        specs,
        solver_names,
        data_source,
        real_data_path,
        real_city_pair,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = QueueWriter(self._append_log)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                run_experiment_suite(
                    specs, solver_names, timestamp,
                    data_source=data_source,
                    real_data_path=real_data_path or None,
                    real_city_pair=real_city_pair,
                )
        except Exception as exc:
            self._append_log(f"\n[错误] {exc}\n")
        finally:
            self.running = False
            self.after(0, lambda: self.run_button.configure(state="normal"))
            self._append_log("\n运行线程已结束。\n")


def launch_gui():
    ExperimentApp().mainloop()
