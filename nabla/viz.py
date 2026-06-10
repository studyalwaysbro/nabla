"""Dependency-free computation graph visualization helpers."""

from __future__ import annotations


def _dot_label(tensor) -> str:
    op = tensor._op if tensor._op else "leaf"
    return f"shape={tensor.data.shape}\\nop={op}".replace('"', '\\"')


def draw_dot(tensor) -> str:
    """Return Graphviz DOT source for the grad-enabled ancestors of ``tensor``."""
    nodes = set()
    edges = set()

    def build(node):
        if node in nodes:
            return
        nodes.add(node)
        for child in node._prev:
            edges.add((child, node))
            build(child)

    build(tensor)

    ordered_nodes = sorted(nodes, key=id)
    node_ids = {node: f"n{id(node)}" for node in ordered_nodes}
    ordered_edges = sorted(edges, key=lambda edge: (id(edge[0]), id(edge[1])))

    lines = ["digraph G {", "  rankdir=LR;"]
    for node in ordered_nodes:
        lines.append(f'  {node_ids[node]} [label="{_dot_label(node)}"];')
    for src, dst in ordered_edges:
        lines.append(f"  {node_ids[src]} -> {node_ids[dst]};")
    lines.append("}")
    return "\n".join(lines)
