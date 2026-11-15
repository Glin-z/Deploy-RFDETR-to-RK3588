import re
import sys
from pathlib import Path

from rknn.api import RKNN

HERE = Path(__file__).resolve().parent
DEFAULT_ONNX = HERE.parent / "Path" / "rf-detr-small-nogs.onnx"
LOG_PATH = HERE / "rknn-convert.log"

TARGET_PLATFORM = "rk3588"

MEAN = [[123.675, 116.28, 103.53]]
STD = [[58.395, 57.12, 57.375]]

INPUT_NAME = "input"
INPUT_SIZE = [1, 3, 512, 512]

DISABLE_RULES = [
    "fuse_matmul_softmax_matmul_to_sdpa",
    "squeeze_nd_elementwise_op",
    "squeeze_nd_mul",
]

MAX_ATTEMPTS = 3


def rules_from_log() -> list[str]:
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
        except Exception as exc:
            print(f"[警告] 构建失败：{str(exc).strip().splitlines()[-1]}")
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
