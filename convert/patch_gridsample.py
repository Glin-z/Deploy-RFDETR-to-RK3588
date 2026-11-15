# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""把 ONNX 模型里的 ``GridSample`` 改写成 RKNN 支持的算子组合。

RF-DETR 的解码器用可变形注意力做跨注意力采样，导出成 ONNX 后是
``GridSample``（bilinear / align_corners=0 / padding_mode=zeros）。这个算子
在 RK3588 的 librknnrt 里没有 CPU 实现，转换能过、上板一跑就报
``Unsupport CPU op: GridSample`` 然后段错误，所以上板前要把它拆掉。

改写后的子图只用 ``Slice / Reshape / Transpose / Add / Sub / Mul / Floor /
Clip / Cast / Unsqueeze / GatherElements``，用双线性插值的定义手算四个角点：

    pixel = (grid + 1) * size / 2 - 0.5          # align_corners=0
    x0 = floor(pixel_x), fx = pixel_x - x0
    out = Σ w_i * x[..., y_i, x_i]               # 越界角点按 0 处理

用法示例：

    python patch_gridsample.py
    python patch_gridsample.py --onnx rf-detr-small.onnx --output rf-detr-small-nogs.onnx

默认读 ``../rf-detr-onnx-main/rf-detr-small.onnx``，默认写到同目录的
``*-nogs.onnx``（nogs = no grid sample）。改完的模型可以直接拿去
``onnx2rknn-simplify.py`` 转换。

返回码：0 成功，1 失败（模型里没有 GridSample、输入 shape 不固定等）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

CONVERT_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = CONVERT_DIR.parent / "rf-detr-onnx-main" / "rf-detr-small.onnx"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class PatchError(RuntimeError):
    """模型结构不符合预期时抛出。"""


class _Builder:
    """按顺序累积节点和常量，名字统一加 ``gs_`` 前缀避免与图内名字冲突。"""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._seq = 0

    def const(self, value: object, dtype: object = np.float32) -> str:
        """加一个常量输入，返回它的名字。"""
        self._seq += 1
        name = f"gs_c{self._seq}"
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def scalar(self, value: float, dtype: object = np.float32) -> str:
        """加一个 0 维标量常量，``Clip`` 的 min/max 必须是 rank-0。"""
        self._seq += 1
        name = f"gs_s{self._seq}"
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype).reshape(()), name))
        return name

    def op(self, op_type: str, inputs: list[str], outputs: list[str] | None = None, **attrs: object) -> str:
        """加一个节点，返回（第一个）输出的名字。"""
        self._seq += 1
        out = outputs[0] if outputs else f"gs_{op_type.lower()}{self._seq}"
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def binary(self, op_type: str, a: str, b: str) -> str:
        """二元算子（Add/Sub/Mul/Div）的简写。"""
        return self.op(op_type, [a, b])


def _static_shape(shape: list[int], name: str) -> list[int]:
    """确认某个张量的形状全为静态，返回该形状。"""
    if not shape or any(dim <= 0 for dim in shape):
        raise PatchError(f"张量 {name} 的形状不是全静态：{shape}，请先用 --input-size 固定输入再导出 ONNX")
    return shape


def _rewrite_one(builder: _Builder, node: onnx.NodeProto, shapes: dict[str, list[int]]) -> None:
    """把一个 GridSample 节点展开成等价的普通算子子图。

    Args:
        builder: 负责累积新节点和常量。
        node: 原始的 GridSample 节点（只支持 bilinear / align_corners=0 / zeros）。
        shapes: 形状表，来自 onnx.shape_inference。
    """
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("mode", b"bilinear") != b"bilinear" or attrs.get("align_corners", 0):
        raise PatchError(f"{node.name}: 只支持 bilinear + align_corners=0，实际是 {attrs}")

    x_name, grid_name = node.input
    n_, c_, h_, w_ = _static_shape(shapes[x_name], x_name)
    gn, gh, gw, _ = _static_shape(shapes[grid_name], grid_name)
    if gn != n_ or gw != 2:
        raise PatchError(f"{node.name}: grid 形状 {gn, gh, gw} 与特征图 {n_, c_, h_, w_} 对不上")
    points = gh * gw
    hw = h_ * w_

    # --- grid 拆成 gx / gy，各是 [N, points]
    gx4 = builder.op(
        "Slice", [grid_name, builder.const([0], np.int64), builder.const([1], np.int64), builder.const([3], np.int64)]
    )
    gy4 = builder.op(
        "Slice", [grid_name, builder.const([1], np.int64), builder.const([2], np.int64), builder.const([3], np.int64)]
    )
    flat_shape = builder.const([gn, points], np.int64)
    gx = builder.op("Reshape", [gx4, flat_shape])
    gy = builder.op("Reshape", [gy4, flat_shape])

    # --- 归一化坐标 → 像素坐标，align_corners=0 时 (g+1)*size/2 - 0.5
    xf = builder.binary("Add", builder.binary("Mul", gx, builder.const([w_ / 2.0])), builder.const([(w_ - 1) / 2.0]))
    yf = builder.binary("Add", builder.binary("Mul", gy, builder.const([h_ / 2.0])), builder.const([(h_ - 1) / 2.0]))
    x0 = builder.op("Floor", [xf])
    y0 = builder.op("Floor", [yf])
    one = builder.const([1.0])
    fx = builder.op("Unsqueeze", [builder.binary("Sub", xf, x0), builder.const([2], np.int64)])
    fy = builder.op("Unsqueeze", [builder.binary("Sub", yf, y0), builder.const([2], np.int64)])
    ifx = builder.op(
        "Unsqueeze", [builder.binary("Sub", one, builder.binary("Sub", xf, x0)), builder.const([2], np.int64)]
    )
    ify = builder.op(
        "Unsqueeze", [builder.binary("Sub", one, builder.binary("Sub", yf, y0)), builder.const([2], np.int64)]
    )

    # --- 特征图按空间维展开成 [N, H*W, C]，这样一次 GatherElements 就能取一个像素的全部通道
    xt = builder.op("Transpose", [x_name], perm=[0, 2, 3, 1])
    xflat = builder.op("Reshape", [xt, builder.const([n_, hw, c_], np.int64)])
    ones3 = builder.const(np.ones((n_, points, c_), np.float32))
    one_f = builder.const([1.0])
    bound_lo = builder.scalar(0.0)
    bound_hi = builder.scalar(1.0)

    def corner(xs: str, ys: str) -> str:
        """取一个角点：索引夹到合法范围，越界的位置用 0 权重屏蔽掉。"""
        idx = builder.binary("Add", builder.binary("Mul", ys, builder.const([float(w_)])), xs)
        idx = builder.op("Clip", [idx, builder.scalar(0.0), builder.scalar(float(hw - 1))])
        idx3 = builder.op(
            "Cast",
            [builder.binary("Mul", builder.op("Unsqueeze", [idx, builder.const([2], np.int64)]), ones3)],
            to=TensorProto.INT32,
        )
        val = builder.op("GatherElements", [xflat, idx3], axis=1)
        # 整数坐标 v 合法 ⇔ 0 <= v <= size-1，用 Clip 表达：clip(v+1,0,1) * clip(size-v,0,1)
        xok = builder.binary(
            "Mul",
            builder.op("Clip", [builder.binary("Add", xs, one_f), bound_lo, bound_hi]),
            builder.op("Clip", [builder.binary("Sub", builder.const([float(w_)]), xs), bound_lo, bound_hi]),
        )
        yok = builder.binary(
            "Mul",
            builder.op("Clip", [builder.binary("Add", ys, one_f), bound_lo, bound_hi]),
            builder.op("Clip", [builder.binary("Sub", builder.const([float(h_)]), ys), bound_lo, bound_hi]),
        )
        mask = builder.op(
            "Mul", [builder.op("Unsqueeze", [builder.binary("Mul", xok, yok), builder.const([2], np.int64)]), ones3]
        )
        return builder.binary("Mul", val, mask)

    x1 = builder.binary("Add", x0, one)
    y1 = builder.binary("Add", y0, one)
    acc = builder.binary("Mul", corner(x0, y0), builder.binary("Mul", ifx, ify))
    acc = builder.binary("Add", acc, builder.binary("Mul", corner(x1, y0), builder.binary("Mul", fx, ify)))
    acc = builder.binary("Add", acc, builder.binary("Mul", corner(x0, y1), builder.binary("Mul", ifx, fy)))
    acc = builder.binary("Add", acc, builder.binary("Mul", corner(x1, y1), builder.binary("Mul", fx, fy)))

    # --- [N, points, C] → [N, C, gh, gw]
    out_t = builder.op("Transpose", [acc], perm=[0, 2, 1])
    builder.op("Reshape", [out_t, builder.const([n_, c_, gh, gw], np.int64)], outputs=[node.output[0]])


def strip_grid_sample(model: onnx.ModelProto) -> int:
    """把模型里所有 GridSample 节点替换掉，返回替换的节点个数。

    Args:
        model: 待处理的 ONNX 模型（原地修改）。

    Returns:
        被替换的 GridSample 节点个数；0 表示模型里没有这个算子。
    """
    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for value_info in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        tensor = value_info.type.tensor_type
        shapes[value_info.name] = [dim.dim_value for dim in tensor.shape.dim]

    builder = _Builder()
    nodes: list[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "GridSample":
            nodes.append(node)
            continue
        _rewrite_one(builder, node, shapes)
        nodes.extend(builder.nodes)
        builder.nodes = []
        replaced += 1

    if replaced:
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        model.graph.initializer.extend(builder.inits)
        onnx.checker.check_model(model, full_check=False)
    return replaced


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="把 ONNX 模型里的 GridSample 换成 RKNN 支持的算子组合",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="输入 ONNX 模型")
    parser.add_argument("--output", default=None, help="输出 ONNX 模型，默认是输入同目录的 <name>-nogs.onnx")
    parser.add_argument("--force", action="store_true", help="输出文件已存在时覆盖")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表，默认取 sys.argv[1:]。

    Returns:
        进程返回码：0 成功，1 失败。
    """
    args = build_parser().parse_args(argv)
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        print(f"[错误] 找不到 ONNX 模型：{onnx_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve() if args.output else onnx_path.with_name(f"{onnx_path.stem}-nogs.onnx")
    )
    if output_path == onnx_path:
        print("[错误] --output 不能和 --onnx 相同", file=sys.stderr)
        return 1
    if output_path.exists() and not args.force:
        print(f"[错误] 输出已存在：{output_path}（加 --force 覆盖）", file=sys.stderr)
        return 1

    print(f"[1/3] 读取模型 {onnx_path}")
    model = onnx.load(str(onnx_path))
    try:
        replaced = strip_grid_sample(model)
    except PatchError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    if not replaced:
        print("[提示] 模型里没有 GridSample，不需要改写")
        return 0

    print(f"[2/3] 替换了 {replaced} 个 GridSample 节点，共 {len(model.graph.node)} 个节点")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"[3/3] 已保存 {output_path}（{output_path.stat().st_size / 1e6:.1f} MB）")
    print(f"接下来：python onnx2rknn-simplify.py {output_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
