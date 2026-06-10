import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from nabla import Tensor, draw_dot


def test_draw_dot_contains_labels_and_edges():
    x = Tensor([1.0, 2.0])
    y = (x * x).sum()
    dot = draw_dot(y)

    assert dot.startswith("digraph G")
    assert "shape=(2,)\\nop=leaf" in dot
    assert "shape=(2,)\\nop=*" in dot
    assert "shape=()\\nop=sum" in dot
    assert dot.count(" -> ") == 2
