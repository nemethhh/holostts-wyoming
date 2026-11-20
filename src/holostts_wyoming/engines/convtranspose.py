"""Rewrite 1-D ConvTranspose as zero insertion + Conv.

ONNX Runtime's WebGPU EP (onnxruntime-ep-webgpu 0.6.0) computes Holos's strided 1-D ConvTranspose wrongly;
the equivalent Conv is computed correctly.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _attrs(node: onnx.NodeProto) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _supported(node: onnx.NodeProto, weights: dict[str, np.ndarray]) -> bool:
    a = _attrs(node)
    if node.input[1] not in weights or weights[node.input[1]].ndim != 3:
        return False
    if a.get("group", 1) != 1 or a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        return False
    if any(v != 1 for v in a.get("dilations", [1])) or any(v != 0 for v in a.get("output_padding", [0])):
        return False
    return "output_shape" not in a


def rewrite_conv_transpose(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    weights = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    nodes = []
    for node in graph.node:
        if node.op_type != "ConvTranspose" or not _supported(node, weights):
            nodes.append(node)
            continue
        a = _attrs(node)
        w = weights[node.input[1]]  # [Cin, Cout, K]
        k = w.shape[2]
        stride = a.get("strides", [1])[0]
        pad_begin, pad_end = a.get("pads", [0, 0])
        p = f"{node.name or node.output[0]}_ct"
        conv_w = np.ascontiguousarray(np.flip(w, axis=2).transpose(1, 0, 2))
        graph.initializer.extend(
            [
                numpy_helper.from_array(conv_w, f"{p}_w"),
                numpy_helper.from_array(np.array([3], np.int64), f"{p}_axis"),
                numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, stride - 1], np.int64), f"{p}_pads"),
                numpy_helper.from_array(np.array([0, 0, -1], np.int64), f"{p}_shape"),
                numpy_helper.from_array(np.array([0], np.int64), f"{p}_start"),
                numpy_helper.from_array(np.array([-(stride - 1) if stride > 1 else np.iinfo(np.int64).max], np.int64), f"{p}_end"),
                numpy_helper.from_array(np.array([2], np.int64), f"{p}_slice_axis"),
            ]
        )
        conv_inputs = [f"{p}_up", f"{p}_w"] + ([node.input[2]] if len(node.input) > 2 and node.input[2] else [])
        nodes += [
            helper.make_node("Unsqueeze", [node.input[0], f"{p}_axis"], [f"{p}_x4"]),
            helper.make_node("Pad", [f"{p}_x4", f"{p}_pads"], [f"{p}_padded"]),
            helper.make_node("Reshape", [f"{p}_padded", f"{p}_shape"], [f"{p}_flat"]),
            helper.make_node("Slice", [f"{p}_flat", f"{p}_start", f"{p}_end", f"{p}_slice_axis"], [f"{p}_up"]),
            helper.make_node(
                "Conv", conv_inputs, list(node.output), name=f"{p}_conv",
                kernel_shape=[k], pads=[k - 1 - pad_begin, k - 1 - pad_end],
            ),
        ]
    del graph.node[:]
    graph.node.extend(nodes)
    return model
