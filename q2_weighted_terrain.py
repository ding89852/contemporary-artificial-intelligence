from __future__ import annotations

import ast
import heapq
import itertools
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

Coord = Tuple[int, int]

DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

def cell_cost(cell: object) -> Optional[float]:
    if cell is None:
        return None
    if isinstance(cell, bool):
        return 1.0
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip().upper()
    if text == "E":
        return 0.0
    if text.isdigit():
        return float(text)
    return 1.0


def solve_weighted_terrain(
    start: Sequence[int],
    goal: Sequence[int],
    grid: Sequence[Sequence[object]],
) -> int:
    if not grid or not grid[0]:
        return -1

    rows, cols = len(grid), len(grid[0])
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    costs = [[cell_cost(cell) for cell in row] for row in grid]

    def inside(node: Coord) -> bool:
        return 0 <= node[0] < rows and 0 <= node[1] < cols

    if not inside(start) or not inside(goal):
        return -1
    if costs[start[0]][start[1]] is None or costs[goal[0]][goal[1]] is None:
        return -1

    goal_cost = costs[goal[0]][goal[1]]
    others = [
        value
        for r, row in enumerate(costs)
        for c, value in enumerate(row)
        if value is not None and (r, c) != goal
    ]
    minimum = min(others) if others else goal_cost

    def heuristic(node: Coord) -> float:
        distance = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        if distance == 0:
            return 0.0
        return (distance - 1) * minimum + min(minimum, goal_cost)

    counter = itertools.count()
    g_score: Dict[Coord, float] = {start: 0.0}
    expanded_at: Dict[Coord, float] = {}
    frontier: List[Tuple[float, float, int, Coord, float]] = []

    def push(node: Coord, g: float) -> None:
        h = heuristic(node)
        heapq.heappush(frontier, (g + h, h, next(counter), node, g))

    push(start, 0.0)

    while frontier:
        _, _, _, node, g = heapq.heappop(frontier)
        if g > g_score.get(node, float("inf")):
            continue
        if node in expanded_at and expanded_at[node] <= g:
            continue
        expanded_at[node] = g

        if node == goal:
            return int(g)

        for dr, dc in DIRECTIONS:
            neighbour = (node[0] + dr, node[1] + dc)
            if not inside(neighbour):
                continue
            step = costs[neighbour[0]][neighbour[1]]
            if step is None:
                continue
            tentative = g + step
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                push(neighbour, tentative)

    return -1


DEFAULT_INPUT = pathlib.Path(__file__).resolve().parent / "inputs" / "q2_weighted.txt"


def _literal(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ast.literal_eval(re.sub(r"(?<![\w\"'])\b([SE])\b(?![\w\"'])", r'"\1"', value))


def load_input(
    path: str | pathlib.Path,
) -> Tuple[Tuple[int, int], Tuple[int, int], List[List[object]]]:
    lines = [line for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = _literal(lines[0])
    goal = _literal(lines[1])
    grid = _literal("\n".join(lines[2:]))
    return tuple(start), tuple(goal), grid


def main(argv: Sequence[str]) -> None:
    path = argv[1] if len(argv) > 1 else DEFAULT_INPUT
    start, goal, grid = load_input(path)
    print(solve_weighted_terrain(start, goal, grid))


if __name__ == "__main__":
    main(sys.argv)
