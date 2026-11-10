from __future__ import annotations

import heapq
import itertools
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Coord = Tuple[int, int]

TREASURE_MAP: List[List[str]] = []
AGENT_POSITIONS: Dict[str, Coord] = {}
TREASURE_POSITIONS: Dict[str, Coord] = {}
TREASURE_VALUES: Dict[str, int] = {}
ENTITY_ORDER: Tuple[str, ...] = ()
TREASURE_ORDER: Tuple[str, ...] = ()
PLAYERS: Tuple[str, str] = ("A", "B")
ENTITY_POSITIONS: Dict[str, Coord] = {}

DEFAULT_INPUT = pathlib.Path(__file__).resolve().parent / "inputs" / "q4_treasure.json"


def load_problem(path: str | pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def configure(problem: dict) -> None:
    global TREASURE_MAP, AGENT_POSITIONS, TREASURE_POSITIONS, TREASURE_VALUES
    global ENTITY_ORDER, TREASURE_ORDER, ENTITY_POSITIONS, _MATRIX_CACHE, _ENTITY_INDEX
    TREASURE_MAP = [list(row) for row in problem["map"]]
    AGENT_POSITIONS = {
        name: (int(position[0]), int(position[1])) for name, position in problem["agents"].items()
    }
    TREASURE_POSITIONS = {
        name: (int(item["position"][0]), int(item["position"][1]))
        for name, item in problem["treasures"].items()
    }
    TREASURE_VALUES = {name: int(item["value"]) for name, item in problem["treasures"].items()}
    TREASURE_ORDER = tuple(sorted(TREASURE_POSITIONS, key=lambda name: int(name[1:])))
    ENTITY_ORDER = tuple(AGENT_POSITIONS) + TREASURE_ORDER
    ENTITY_POSITIONS = {**AGENT_POSITIONS, **TREASURE_POSITIONS}
    _MATRIX_CACHE = None
    _ENTITY_INDEX = {position: name for name, position in ENTITY_POSITIONS.items()}

DIRECTIONS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEG_INF = float("-inf")
POS_INF = float("inf")


def cell_cost(cell: object) -> Optional[float]:
    if cell is None:
        return None
    if isinstance(cell, (int, float)) and not isinstance(cell, bool):
        return float(cell)
    text = str(cell).strip().upper()
    if text in ("X", "#"):
        return None
    if text.isdigit():
        return float(text)
    return 1.0


def astar_cost(
    start: Sequence[int],
    goal: Sequence[int],
    grid: Optional[Sequence[Sequence[object]]] = None,
) -> int:
    grid = TREASURE_MAP if grid is None else grid
    rows, cols = len(grid), len(grid[0])
    costs = [[cell_cost(cell) for cell in row] for row in grid]
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if costs[start[0]][start[1]] is None or costs[goal[0]][goal[1]] is None:
        return -1

    def heuristic(node: Coord) -> int:
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    counter = itertools.count()
    g_score: Dict[Coord, float] = {start: 0.0}
    closed = set()
    frontier = [(heuristic(start), heuristic(start), next(counter), start)]
    while frontier:
        _, _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            return int(g_score[node])
        for dr, dc in DIRECTIONS:
            neighbour = (node[0] + dr, node[1] + dc)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols):
                continue
            step = costs[neighbour[0]][neighbour[1]]
            if step is None:
                continue
            tentative = g_score[node] + step
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                heapq.heappush(
                    frontier,
                    (tentative + heuristic(neighbour), heuristic(neighbour), next(counter), neighbour),
                )
    return -1


_MATRIX_CACHE: Optional[List[List[int]]] = None
_ENTITY_INDEX: Dict[Coord, str] = {}


def distance_matrix(order: Optional[Sequence[str]] = None) -> List[List[int]]:
    global _MATRIX_CACHE
    names = tuple(order) if order is not None else ENTITY_ORDER
    if names == ENTITY_ORDER and _MATRIX_CACHE is not None:
        return _MATRIX_CACHE
    size = len(names)
    matrix = [[0] * size for _ in range(size)]
    for row, source in enumerate(names):
        for column in range(row + 1, size):
            target = names[column]
            cost = astar_cost(ENTITY_POSITIONS[source], ENTITY_POSITIONS[target])
            if cost < 0:
                raise ValueError(f"{source} 到 {target} 不可达")
            matrix[row][column] = matrix[column][row] = cost
    if names == ENTITY_ORDER:
        _MATRIX_CACHE = matrix
    return matrix


def matrix_to_markdown(order: Optional[Sequence[str]] = None) -> str:
    names = list(order) if order is not None else list(ENTITY_ORDER)
    matrix = distance_matrix(names)
    header = "| 从 \\ 到 | " + " | ".join(names) + " |"
    divider = "| --- | " + " | ".join("---" for _ in names) + " |"
    rows = [
        "| " + names[i] + " | " + " | ".join(str(matrix[i][j]) for j in range(len(names))) + " |"
        for i in range(len(names))
    ]
    return "\n".join([header, divider, *rows])


def gain(anchor: object, treasure: str, order: Optional[Sequence[str]] = None) -> int:
    matrix = distance_matrix(order)
    if isinstance(anchor, str):
        source = anchor
    else:
        position = (int(anchor[0]), int(anchor[1]))
        if position not in _ENTITY_INDEX:
            raise ValueError(f"坐标 {position} 不是任何实体的位置")
        source = _ENTITY_INDEX[position]
    return TREASURE_VALUES[treasure] - matrix[ENTITY_ORDER.index(source)][ENTITY_ORDER.index(treasure)]


@dataclass(frozen=True)
class GameState:

    anchors: Tuple[str, str]
    remaining: Tuple[str, ...]
    turn: int
    score: Tuple[int, int]

    @property
    def value(self) -> int:
        return self.score[0] - self.score[1]


@dataclass
class SearchStats:
    nodes_visited: int = 0
    evaluations: int = 0
    cutoffs: int = 0
    pruned_moves: int = 0


@dataclass
class GameOutcome:
    best_target: str
    best_value: int
    best_gain: int
    stats: SearchStats
    pv: List[Tuple[str, str, int]]


def initial_state() -> GameState:
    return GameState(("A", "B"), TREASURE_ORDER, 0, (0, 0))


def _apply(state: GameState, target: str, gain_value: int) -> GameState:
    anchors = list(state.anchors)
    anchors[state.turn] = target
    score = list(state.score)
    score[state.turn] += gain_value
    remaining = tuple(name for name in state.remaining if name != target)
    return GameState(tuple(anchors), remaining, 1 - state.turn, tuple(score))


def _moves(state: GameState, matrix: List[List[int]]) -> List[Tuple[str, int]]:
    source = ENTITY_ORDER.index(state.anchors[state.turn])
    return [
        (target, TREASURE_VALUES[target] - matrix[source][ENTITY_ORDER.index(target)])
        for target in state.remaining
    ]


def _search(
    state: GameState,
    depth_left: int,
    pruning: bool,
    alpha: float,
    beta: float,
    stats: SearchStats,
    matrix: List[List[int]],
) -> Tuple[float, Optional[str], Optional[int]]:
    stats.nodes_visited += 1
    if depth_left == 0 or not state.remaining:
        stats.evaluations += 1
        return state.value, None, None

    maximizing = state.turn == 0
    best_value = NEG_INF if maximizing else POS_INF
    best_target: Optional[str] = None
    best_gain: Optional[int] = None
    moves = _moves(state, matrix)

    for position, (target, gain_value) in enumerate(moves):
        child = _apply(state, target, gain_value)
        value, _, _ = _search(child, depth_left - 1, pruning, alpha, beta, stats, matrix)
        if maximizing:
            if value > best_value:
                best_value, best_target, best_gain = value, target, gain_value
            if pruning:
                if best_value >= beta:
                    skipped = len(moves) - position - 1
                    if skipped:
                        stats.cutoffs += 1
                        stats.pruned_moves += skipped
                    break
                alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value, best_target, best_gain = value, target, gain_value
            if pruning:
                if best_value <= alpha:
                    skipped = len(moves) - position - 1
                    if skipped:
                        stats.cutoffs += 1
                        stats.pruned_moves += skipped
                    break
                beta = min(beta, best_value)
    return best_value, best_target, best_gain


def solve_game(
    depth: int,
    algorithm: str = "alpha_beta",
    compute_pv: bool = True,
) -> GameOutcome:
    if depth < 1:
        raise ValueError("搜索深度至少为 1")
    if algorithm not in ("minimax", "alpha_beta"):
        raise ValueError("algorithm 只能是 'minimax' 或 'alpha_beta'")
    matrix = distance_matrix()
    stats = SearchStats()
    value, target, gain_value = _search(
        initial_state(), depth, algorithm == "alpha_beta", NEG_INF, POS_INF, stats, matrix
    )
    if target is None:
        raise RuntimeError("没有可选的宝藏")
    pv = principal_variation(depth, matrix) if compute_pv else []
    return GameOutcome(target, int(value), int(gain_value), stats, pv)


def principal_variation(depth: int, matrix: Optional[List[List[int]]] = None) -> List[Tuple[str, str, int]]:
    table = matrix if matrix is not None else distance_matrix()
    state = initial_state()
    stats = SearchStats()
    line: List[Tuple[str, str, int]] = []
    for _ in range(depth):
        if not state.remaining:
            break
        _, target, gain_value = _search(
            state, depth - len(line), False, NEG_INF, POS_INF, stats, table
        )
        if target is None:
            break
        line.append((PLAYERS[state.turn], target, int(gain_value)))
        state = _apply(state, target, gain_value)
    return line


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = sys.argv if argv is None else argv
    path = args[1] if len(args) > 1 else DEFAULT_INPUT
    problem = load_problem(path)
    configure(problem)
    print("最小移动代价矩阵（" + ", ".join(ENTITY_ORDER) + "）:")
    print(matrix_to_markdown())
    print()
    for depth in problem.get("depths", [2, 4]):
        for label, algorithm in (("Minimax", "minimax"), ("Alpha-Beta", "alpha_beta")):
            outcome = solve_game(depth, algorithm=algorithm)
            print(
                f'H = {depth}  {label:<10} => '
                f'{{"best_target": "{outcome.best_target}", "best_value": {outcome.best_value}}}'
                f"   访问 {outcome.stats.nodes_visited} 个节点"
            )
        print(f"       主变例: {principal_variation(depth)}")


configure(load_problem(DEFAULT_INPUT))


if __name__ == "__main__":
    main()
