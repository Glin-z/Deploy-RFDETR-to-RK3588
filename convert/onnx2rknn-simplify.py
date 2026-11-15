"""RF-DETR ONNX -> RKNN 精简版（rk3588 / FP16）。

只保留转换成功所必需的动作：config -> load_onnx -> build -> export_rknn。
完整版（量化、校准集、精度对比、参数开关）见 onnx2rknn.py。

用法：
    python onnx2rknn-simplify.py                     # 转默认的 rf-detr-small.onnx
    python onnx2rknn-simplify.py rf-detr-small-nogs.onnx   # 转去掉 GridSample 的模型

上板跑之前请先执行 ``python patch_gridsample.py``，用它输出的 ``*-nogs.onnx``
来转换：原模型里的 GridSample 在 RK3588 的 librknnrt 里没有 CPU 实现，转出来
的 rknn 能加载，但一推理就报 ``Unsupport CPU op: GridSample`` 然后段错误。
"""

import re
import sys
from pathlib import Path

from rknn.api import RKNN

# ---- 路径：默认模型，换模型时用命令行参数或改这里 --------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE.parent / "RFDETR" / "rf-detr-small-nogs.onnx"
LOG_PATH = HERE / "rknn-convert.log"

# ---- 转换参数 ------------------------------------------------------------
TARGET_PLATFORM = "rk3588"

# 训练和 ONNX 推理都是 (x / 255 - mean) / std，所以这里传 0~255 尺度的 mean/std，
# 由 rknn 在板端前处理里完成减均值除方差。用错会把得分压到 0 附近。
MEAN = [[123.675, 116.28, 103.53]]
STD = [[58.395, 57.12, 57.375]]

# 模型输入：静态 1x3x512x512。显式传 inputs/input_size_list 会让工具链按这个形状裁剪 ONNX，
# 这是实测能通过 build 的走法（不传时 fuse_ops 会踩到 rules/dims.py 的另一个 bug）。
INPUT_NAME = "input"
INPUT_SIZE = [1, 3, 512, 512]

# rknn-toolkit2 2.5.0 的 fuse_ops 里有两条规则处理本模型的子图时会抛工具链内部异常：
#   fuse_matmul_softmax_matmul_to_sdpa -> ValueError: inputs or 'outputs' must be set!
#   squeeze_nd_elementwise_op          -> AttributeError: 'IRGraph' object has no attribute 'node_use_target_const'
# 注册名和报错栈里的函数名不一定一样，写错的名字会被工具链忽略，所以多列几个没有副作用。
DISABLE_RULES = [
    "fuse_matmul_softmax_matmul_to_sdpa",
    "squeeze_nd_elementwise_op",
    "squeeze_nd_mul",
]

MAX_ATTEMPTS = 3


def rules_from_log() -> list[str]:
    """从日志里抓工具链自己建议禁用的规则名（形如 disable_rules=['xxx']）。

    Returns:
        去重后的规则名列表，日志里没有这类提示时为空。
    """
    try:
        text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    names: list[str] = []
    for group in re.findall(r"disable_rules=\[([^\]]*)\]", text):
        for name in re.findall(r"['\"]([^'\"]+)['\"]", group):
            if name not in names:
                names.append(name)
    return names


def convert_once(onnx_path: Path, rknn_path: Path, rules: list[str]) -> None:
    """用给定的 disable_rules 跑一遍转换，失败时由调用方处理异常。

    Args:
        onnx_path: 要转换的 ONNX 模型。
        rknn_path: 导出的 rknn 模型路径。
        rules: 要交给 ``rknn.config(disable_rules=...)`` 的规则名。
    """
    rknn = RKNN(verbose=True, verbose_file=str(LOG_PATH))
    try:
        rknn.config(
            target_platform=TARGET_PLATFORM,
            mean_values=MEAN,
            std_values=STD,
            disable_rules=rules,
        )
        rknn.load_onnx(model=str(onnx_path), inputs=[INPUT_NAME], input_size_list=[INPUT_SIZE])
        rknn.build(do_quantization=False)
        rknn.export_rknn(str(rknn_path))
    finally:
        rknn.release()


def main() -> int:
    """执行转换，返回进程退出码（0 成功，1 失败）。"""
    onnx_path = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_ONNX
    rknn_path = HERE / f"{onnx_path.stem}.rknn"
    if not onnx_path.is_file():
        print(f"[错误] 找不到 ONNX 模型：{onnx_path}")
        return 1

    rules = list(DISABLE_RULES)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[尝试 {attempt}/{MAX_ATTEMPTS}] {onnx_path.name} {TARGET_PLATFORM} / FP16，disable_rules={rules}")
        try:
            convert_once(onnx_path, rknn_path, rules)
        except Exception as exc:  # 工具链内部异常五花八门，统一按“换一组规则重试”处理
            print(f"[警告] 构建失败：{str(exc).strip().splitlines()[-1]}")
            # 优先用日志里工具链给出的建议名；没有建议就说明是没见过的规则 bug，放弃
            added = [name for name in rules_from_log() if name not in rules]
            if not added or attempt == MAX_ATTEMPTS:
                print(f"[错误] 转换失败，完整日志见 {LOG_PATH}")
                return 1
            rules.extend(added)
            continue

        size_mb = rknn_path.stat().st_size / (1024 * 1024)
        print(f"[完成] {rknn_path}（{size_mb:.1f} MB）")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
