from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

NEG_INF = float("-inf")
POS_INF = float("inf")


class TreeNode:

    __slots__ = ("value", "children", "is_maximizing_player")

    def __init__(
        self,
        value: Optional[float] = None,
        children: Optional[Sequence["TreeNode"]] = None,
        is_maximizing_player: Optional[bool] = None,
    ) -> None:
        self.value = value
        self.children: List["TreeNode"] = list(children) if children else []
        self.is_maximizing_player = is_maximizing_player

    def __repr__(self) -> str:
        if self.children:
            return f"TreeNode(children={len(self.children)})"
        return f"TreeNode(value={self.value})"


@dataclass
class SearchOutcome:

    value: float
    nodes_visited: int = 0
    leaves_evaluated: int = 0
    cutoffs: int = 0


class _Stats:
    __slots__ = ("nodes_visited", "leaves_evaluated", "cutoffs")

    def __init__(self) -> None:
        self.nodes_visited = 0
        self.leaves_evaluated = 0
        self.cutoffs = 0


def from_dict(data: object) -> TreeNode:
    if isinstance(data, TreeNode):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"无法从 {type(data).__name__} 构建博弈树")
    if "tree_structure" in data:
        return from_dict(data["tree_structure"])
    turn = data.get("is_maximizing_player")
    if "children" not in data:
        if "value" not in data:
            raise ValueError("叶节点必须提供 value 字段")
        return TreeNode(value=data["value"], is_maximizing_player=turn)
    return TreeNode(children=[from_dict(child) for child in data["children"]], is_maximizing_player=turn)


def count_nodes(node: TreeNode) -> int:
    return 1 + sum(count_nodes(child) for child in node.children)


def _child_turn(child: TreeNode, parent_is_max: bool) -> bool:
    if child.is_maximizing_player is None:
        return not parent_is_max
    return bool(child.is_maximizing_player)


def _minimax(node: TreeNode, is_maximizing: bool, stats: _Stats) -> float:
    stats.nodes_visited += 1
    if not node.children:
        stats.leaves_evaluated += 1
        return float(node.value)
    values = [
        _minimax(child, _child_turn(child, is_maximizing), stats) for child in node.children
    ]
    return max(values) if is_maximizing else min(values)


def minimax(node: TreeNode, is_maximizing: bool = True) -> SearchOutcome:
    stats = _Stats()
    value = _minimax(node, is_maximizing, stats)
    return SearchOutcome(value, stats.nodes_visited, stats.leaves_evaluated, 0)


def _alpha_beta(
    node: TreeNode,
    is_maximizing: bool,
    alpha: float,
    beta: float,
    stats: _Stats,
) -> float:
    stats.nodes_visited += 1
    if not node.children:
        stats.leaves_evaluated += 1
        return float(node.value)

    if is_maximizing:
        value = NEG_INF
        for index, child in enumerate(node.children):
            value = max(value, _alpha_beta(child, _child_turn(child, is_maximizing), alpha, beta, stats))
            if value >= beta:
                if index + 1 < len(node.children):
                    stats.cutoffs += 1
                break
            alpha = max(alpha, value)
        return value

    value = POS_INF
    for index, child in enumerate(node.children):
        value = min(value, _alpha_beta(child, _child_turn(child, is_maximizing), alpha, beta, stats))
        if value <= alpha:
            if index + 1 < len(node.children):
                stats.cutoffs += 1
            break
        beta = min(beta, value)
    return value


def alpha_beta(
    node: TreeNode,
    is_maximizing: bool = True,
    alpha: float = NEG_INF,
    beta: float = POS_INF,
) -> SearchOutcome:
    stats = _Stats()
    value = _alpha_beta(node, is_maximizing, alpha, beta, stats)
    return SearchOutcome(value, stats.nodes_visited, stats.leaves_evaluated, stats.cutoffs)


def solve_tree(tree: object, algorithm: str = "alpha_beta") -> float:
    node = tree if isinstance(tree, TreeNode) else from_dict(tree)
    is_maximizing = True if node.is_maximizing_player is None else bool(node.is_maximizing_player)
    if algorithm == "minimax":
        value = minimax(node, is_maximizing).value
    elif algorithm == "alpha_beta":
        value = alpha_beta(node, is_maximizing).value
    else:
        raise ValueError("algorithm 只能是 'minimax' 或 'alpha_beta'")
    return int(value) if float(value).is_integer() else value


DEFAULT_INPUT = pathlib.Path(__file__).resolve().parent / "inputs" / "q3_tree.json"


def load_tree(path: str | pathlib.Path) -> TreeNode:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return from_dict(data)


def main(argv: Sequence[str]) -> None:
    path = argv[1] if len(argv) > 1 else DEFAULT_INPUT
    print(solve_tree(load_tree(path)))


if __name__ == "__main__":
    main(sys.argv)
