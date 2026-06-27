"""customtkinter 可视化实验工具。

界面只负责读取用户输入、预览计划和展示运行日志。模型参数输入框根据
DeliveryConfig 动态生成，因此在 config.py 增加 dataclass 字段后无需修改本文件。
"""

import contextlib
import io
import threading
from datetime import datetime
from typing import List

import customtkinter as ctk

from experiment_core import (
    ExperimentPlan,
    build_specs,
    get_sensitivity_parameters,
    levels_to_text,
    parse_parameter_levels,
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
    """仿真实验主窗口。"""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("城际配送系统仿真实验平台")
        self.geometry("1250x820")
        self.minsize(1080, 720)

        self.plan = ExperimentPlan()
        self.sensitivity_parameters = get_sensitivity_parameters()
        self.running = False
        self.scenario_vars = {}
        self.solver_vars = {}
        self.fields = {}
        self.sensitivity_fields = {}

        self._build_layout()
        self._refresh_preview()

    def _build_layout(self):
        """搭建窗口左右两栏布局。"""

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="城际配送系统仿真实验平台",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        left_panel = ctk.CTkFrame(self, width=440)
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
        """创建两类实验场景和求解器选择区。"""

        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="实验场景", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 4)
        )
        scenario_labels = {
            "quick": "快速测试 quick",
            "sensitivity": "灵敏度分析 sensitivity",
        }
        for row, (name, label) in enumerate(scenario_labels.items(), start=1):
            variable = ctk.BooleanVar(value=(name == "quick"))
            self.scenario_vars[name] = variable
            ctk.CTkCheckBox(
                frame,
                text=label,
                variable=variable,
                command=self._refresh_preview,
            ).grid(row=row, column=0, sticky="w", padx=12, pady=4)

        solver_start = len(scenario_labels) + 1
        ctk.CTkLabel(frame, text="求解方式", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=solver_start, column=0, sticky="w", padx=12, pady=(14, 4)
        )
        for row, name in enumerate(SOLVER_REGISTRY, start=solver_start + 1):
            variable = ctk.BooleanVar(value=(name == "exact_mip"))
            self.solver_vars[name] = variable
            ctk.CTkCheckBox(
                frame,
                text=get_solver_display_name(name),
                variable=variable,
                command=self._refresh_preview,
            ).grid(row=row, column=0, sticky="w", padx=12, pady=4)

    def _add_entry(self, frame, row, label, value):
        """添加一个带中文标签的输入框，并绑定实时预览。"""

        ctk.CTkLabel(frame, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=8, pady=5
        )
        entry = ctk.CTkEntry(frame)
        entry.insert(0, value)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        entry.bind("<KeyRelease>", lambda _event: self._refresh_preview())
        return entry

    def _build_parameter_panel(self, parent):
        """根据核心层提供的参数清单动态创建灵敏度输入框。

        每个水平都使用 JSON 数组表示，例如 [10,20,30]。字典参数可以写为
        [{"1":10,"2":10},{"1":30,"2":30}]，解析时会自动恢复整数城市编号。
        """

        frame = ctk.CTkScrollableFrame(parent, label_text="参数范围（灵敏度水平使用 JSON 数组）")
        frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        parent.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        fixed_rows = [
            ("seed_count", "每个水平随机种子数", str(self.plan.seed_count)),
            ("time_limit", "单算例时间限制(秒)", str(self.plan.time_limit)),
            ("quick_orders", "快速测试订单数", str(self.plan.quick_orders)),
        ]
        row = 0
        for key, label, value in fixed_rows:
            self.fields[key] = self._add_entry(frame, row, label, value)
            row += 1

        ctk.CTkLabel(
            frame,
            text="单因素灵敏度参数",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(14, 5))
        row += 1

        for parameter in self.sensitivity_parameters:
            levels = self.plan.sensitivity_levels[parameter.key]
            self.sensitivity_fields[parameter.key] = self._add_entry(
                frame,
                row,
                parameter.label,
                levels_to_text(levels),
            )
            row += 1

    def _build_action_panel(self, parent):
        """创建刷新和运行按钮。"""

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
        """创建实验计划预览框和运行日志框。"""

        ctk.CTkLabel(
            parent, text="实验计划预览", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        self.preview = ctk.CTkTextbox(parent, height=280)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            parent, text="运行日志", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=2, column=0, sticky="w", padx=12, pady=(4, 4))
        self.log = ctk.CTkTextbox(parent, height=260)
        self.log.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        parent.grid_rowconfigure(3, weight=1)

    def _selected_scenarios(self) -> List[str]:
        return [name for name, variable in self.scenario_vars.items() if variable.get()]

    def _selected_solvers(self) -> List[str]:
        return [name for name, variable in self.solver_vars.items() if variable.get()]

    def _read_plan(self) -> ExperimentPlan:
        """读取固定参数和全部动态灵敏度参数。"""

        sensitivity_levels = {}
        for parameter in self.sensitivity_parameters:
            sensitivity_levels[parameter.key] = parse_parameter_levels(
                self.sensitivity_fields[parameter.key].get(),
                parameter.base_value,
            )
        return ExperimentPlan(
            seed_count=int(self.fields["seed_count"].get()),
            time_limit=int(self.fields["time_limit"].get()),
            quick_orders=int(self.fields["quick_orders"].get()),
            sensitivity_levels=sensitivity_levels,
        )

    def _refresh_preview(self):
        """解析当前输入并刷新算例预览，不调用求解器。"""

        try:
            plan = self._read_plan()
            solvers = self._selected_solvers()
            specs = build_specs(self._selected_scenarios(), plan)
        except Exception as exc:
            self.preview.delete("1.0", "end")
            self.preview.insert("end", f"参数暂不可解析：{exc}")
            return

        lines = [
            f"算例数量：{len(specs)}",
            f"求解器数量：{len(solvers)}",
            f"预计结果行数：{len(specs) * len(solvers)}",
            "",
            "前 30 个算例：",
        ]
        for spec in specs[:30]:
            sensitivity = (
                f"{spec.sensitivity_parameter}={spec.sensitivity_value}"
                if spec.sensitivity_parameter
                else "环境连通性检查"
            )
            lines.append(
                f"{spec.experiment_id} | {spec.scenario} | orders={spec.num_orders} | "
                f"seed={spec.seed} | {sensitivity}"
            )
        if len(specs) > 30:
            lines.append(f"... 另有 {len(specs) - 30} 个算例未显示")

        self.preview.delete("1.0", "end")
        self.preview.insert("end", "\n".join(lines))

    def _append_log(self, text: str):
        """从后台线程安全地请求追加日志。"""

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
        except Exception as exc:
            self._append_log(f"\n[参数错误] {exc}\n")
            return

        self.running = True
        self.log.delete("1.0", "end")
        self._append_log(f"准备运行 {len(specs)} 个算例，{len(solver_names)} 个求解器。\n")
        threading.Thread(
            target=self._run_worker,
            args=(specs, solver_names),
            daemon=True,
        ).start()

    def _run_worker(self, specs, solver_names):
        """后台执行批量实验，使窗口在求解期间保持可响应。"""

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

    ExperimentApp().mainloop()
