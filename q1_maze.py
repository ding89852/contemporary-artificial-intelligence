from __future__ import annotations

import ast
import heapq
import itertools
import pathlib
import re
import sys
from typing import List, Optional, Sequence, Tuple

Coord = Tuple[int, int]

DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

BLOCKED = frozenset({"X", "#"})


def solve_maze(
    start: Sequence[int],
    goal: Sequence[int],
    grid: Sequence[Sequence[str]],
) -> List[Coord]:
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    def inside(node: Coord) -> bool:
        return 0 <= node[0] < rows and 0 <= node[1] < cols

    def blocked(node: Coord) -> bool:
        return str(grid[node[0]][node[1]]).strip().upper() in BLOCKED

    if not inside(start) or not inside(goal):
        return []
    if blocked(start) or blocked(goal):
        return []

    def heuristic(node: Coord) -> int:
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    counter = itertools.count()
    g_score = {start: 0}
    parent: dict[Coord, Optional[Coord]] = {start: None}
    closed = set()
    frontier = [(heuristic(start), heuristic(start), next(counter), start)]

    while frontier:
        _, _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            path: List[Coord] = []
            current: Optional[Coord] = node
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        for dr, dc in DIRECTIONS:
            neighbour = (node[0] + dr, node[1] + dc)
            if not inside(neighbour) or blocked(neighbour):
                continue
            tentative = g_score[node] + 1
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                parent[neighbour] = node
                heapq.heappush(
                    frontier,
                    (
                        tentative + heuristic(neighbour),
                        heuristic(neighbour),
                        next(counter),
                        neighbour,
                    ),
                )
    return []


DEFAULT_INPUT = pathlib.Path(__file__).resolve().parent / "inputs" / "q1_maze.txt"


def _literal(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ast.literal_eval(re.sub(r"(?<![\w\"'])\b([SEX])\b(?![\w\"'])", r'"\1"', value))


def load_input(path: str | pathlib.Path) -> Tuple[Tuple[int, int], Tuple[int, int], List[List[str]]]:
    lines = [line for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = _literal(lines[0])
    goal = _literal(lines[1])
    grid = _literal("\n".join(lines[2:]))
    return tuple(start), tuple(goal), grid


def main(argv: Sequence[str]) -> None:
    path = argv[1] if len(argv) > 1 else DEFAULT_INPUT
    start, goal, grid = load_input(path)
    print(solve_maze(start, goal, grid))


if __name__ == "__main__":
    main(sys.argv)
