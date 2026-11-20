import numpy as np
import onnx
import onnxruntime as ort
import pytest
from conftest import MODELS
from onnx import TensorProto, helper, numpy_helper

from holostts_wyoming.engines.convtranspose import rewrite_conv_transpose


def conv_transpose_model(cin, cout, k, stride, pad, bias=True):
    rng = np.random.default_rng(0)
    inits = [numpy_helper.from_array(rng.standard_normal((cin, cout, k)).astype(np.float32), "w")]
    inputs = ["x", "w"]
    if bias:
        inits.append(numpy_helper.from_array(rng.standard_normal(cout).astype(np.float32), "b"))
        inputs.append("b")
    node = helper.make_node("ConvTranspose", inputs, ["y"], strides=[stride], pads=[pad, pad], kernel_shape=[k])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, "L"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def run(model, x):
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@pytest.mark.parametrize("cin,cout,k,stride,pad,bias", [(4, 3, 12, 6, 3, True), (5, 2, 10, 5, 2, True), (3, 3, 4, 2, 1, False)])
def test_rewrite_is_equivalent(cin, cout, k, stride, pad, bias):
    model = conv_transpose_model(cin, cout, k, stride, pad, bias)
    x = np.random.default_rng(1).standard_normal((1, cin, 17)).astype(np.float32)
    rewritten = rewrite_conv_transpose(model)
    assert not any(n.op_type == "ConvTranspose" for n in rewritten.graph.node)
    np.testing.assert_allclose(run(rewritten, x), run(model, x), rtol=1e-4, atol=1e-4)


def test_unsupported_conv_transpose_is_left_alone():
    model = conv_transpose_model(2, 2, 4, 2, 1)
    attr = helper.make_attribute("output_padding", [1])
    model.graph.node[0].attribute.append(attr)
    assert rewrite_conv_transpose(model).graph.node[0].op_type == "ConvTranspose"


def captured_decoder_feeds(model_dir):
    from holos_onnx import HolosTTS

    holos = HolosTTS(str(model_dir), streaming=True)
    real = holos.decoder_session
    feeds = []

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def run(self, outputs, f, *args, **kwargs):
            feeds.append(f)
            return real.run(outputs, f, *args, **kwargs)

    holos.decoder_session = Spy()
    list(holos.stream("ʊˈʋʲimknenɔ ˈsʲʋʲitɫɔ.", voice="Speaker_0"))
    return feeds[0]


@pytest.mark.skipif(not (MODELS / "onnx/decoder_window.onnx").is_file(), reason="run `make models-fp32`")
def test_holos_decoder_rewrite_matches():
    model = onnx.load(MODELS / "onnx/decoder_window.onnx")
    rewritten = rewrite_conv_transpose(model)
    assert sum(n.op_type == "ConvTranspose" for n in rewritten.graph.node) == 0
    feeds = captured_decoder_feeds(MODELS / "onnx")
    a = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)
    b = ort.InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)
    for x, y in zip(a, b):
        np.testing.assert_allclose(y, x, rtol=1e-3, atol=1e-4)
