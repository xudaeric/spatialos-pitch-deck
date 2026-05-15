"""Reference planner for composing a rectangular tabletop from cabinet robots.

The planner intentionally solves only the final arrangement problem. It does not
try to prove that every robot can move from its current pose to the target pose;
that second stage is a separate motion-planning problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from math import cos, hypot, isclose, sin
from statistics import median
from time import monotonic
from typing import Iterable


EPS = 1e-9
HEIGHT_TOLERANCE_M = 0.03


class PlanStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    APPROXIMATE = "APPROXIMATE"
    NO_SOLUTION = "NO_SOLUTION"


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class RobotModule:
    id: str
    length_m: float
    width_m: float
    height_m: float
    current_pose: Pose
    top_flat: bool = True
    available: bool = True
    max_load: float = 0.0
    can_rotate_90: bool = True


@dataclass(frozen=True)
class TableRequest:
    target_length_m: float
    target_width_m: float
    target_pose: Pose = Pose(0.0, 0.0, 0.0)
    target_yaw: float = 0.0
    target_height_m: float | None = None
    tolerance_m: float = 0.02
    max_gap_m: float = 0.02
    max_overhang_m: float = 0.03
    time_budget_ms: int = 500
    max_states: int = 50_000


@dataclass(frozen=True)
class Placement:
    robot_id: str
    center_x: float
    center_y: float
    yaw: float
    rect_length: float
    rect_width: float


@dataclass(frozen=True)
class CompositionPlan:
    placements: tuple[Placement, ...] = ()
    score: float = float("inf")
    uncovered_area: float = 0.0
    overhang_area: float = 0.0
    seam_length: float = 0.0
    max_gap: float = 0.0
    status: PlanStatus = PlanStatus.NO_SOLUTION


@dataclass(frozen=True)
class _Candidate:
    robot: RobotModule
    length: float
    width: float
    yaw_offset: float


@dataclass(frozen=True)
class _Rect:
    robot_id: str
    x: float
    y: float
    length: float
    width: float
    yaw_offset: float

    @property
    def right(self) -> float:
        return self.x + self.length

    @property
    def top(self) -> float:
        return self.y + self.width

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.length / 2.0, self.y + self.width / 2.0)


@dataclass(frozen=True)
class _State:
    rects: tuple[_Rect, ...] = ()
    used_ids: frozenset[str] = field(default_factory=frozenset)


def plan_table_composition(request: TableRequest, robots: Iterable[RobotModule]) -> CompositionPlan:
    """Return the best known final tabletop composition for the request."""

    robot_list = tuple(robots)
    candidates = _preprocess_robots(request, robot_list)
    if not candidates:
        return CompositionPlan(status=PlanStatus.NO_SOLUTION)

    unique_area = sum(r.length_m * r.width_m for r in _unique_candidate_robots(candidates))
    if unique_area + _area_tolerance(request) < request.target_length_m * request.target_width_m:
        return CompositionPlan(status=PlanStatus.NO_SOLUTION)

    deadline = monotonic() + request.time_budget_ms / 1000.0
    queue: list[tuple[float, int, _State]] = []
    counter = 0
    initial = _State()
    heappush(queue, (0.0, counter, initial))

    best_complete: CompositionPlan | None = None
    best_partial: tuple[float, _State] | None = None
    seen: set[tuple[tuple[str, float, float, float, float], ...]] = set()
    states_expanded = 0

    while queue and monotonic() < deadline and states_expanded < request.max_states:
        _, _, state = heappop(queue)
        states_expanded += 1

        signature = _state_signature(state)
        if signature in seen:
            continue
        seen.add(signature)

        metrics = _measure_layout(state.rects, request)
        partial_score = _score_metrics(metrics, len(state.rects), _total_distance(state.rects, candidates, request))
        if best_partial is None or partial_score < best_partial[0]:
            best_partial = (partial_score, state)

        if _is_complete(metrics, request):
            plan = _make_plan(state.rects, candidates, request, PlanStatus.FEASIBLE)
            if best_complete is None or _plan_sort_key(plan) < _plan_sort_key(best_complete):
                best_complete = plan
            continue

        anchor = _first_uncovered_frontier_point(state.rects, request)
        if anchor is None:
            continue

        for candidate in _order_candidates_by_fit(candidates, state, anchor, request):
            if candidate.robot.id in state.used_ids:
                continue

            rect = _Rect(
                robot_id=candidate.robot.id,
                x=anchor[0],
                y=anchor[1],
                length=candidate.length,
                width=candidate.width,
                yaw_offset=candidate.yaw_offset,
            )
            if not _within_bounds(rect, request):
                continue
            if any(_overlap(rect, existing) for existing in state.rects):
                continue

            next_state = _State(state.rects + (rect,), state.used_ids | {candidate.robot.id})
            next_metrics = _measure_layout(next_state.rects, request)
            if _largest_gap_dimension(next_metrics) > max(request.max_gap_m, request.tolerance_m) and (
                len(next_state.rects) == len(_unique_candidate_robots(candidates))
            ):
                continue
            counter += 1
            priority = _score_metrics(
                next_metrics,
                len(next_state.rects),
                _total_distance(next_state.rects, candidates, request),
            )
            heappush(queue, (priority, counter, next_state))

    if best_complete is not None:
        return best_complete

    approximate = _plan_approximate_cover(best_partial, candidates, request)
    if approximate is not None:
        return approximate

    return CompositionPlan(status=PlanStatus.NO_SOLUTION)


def _preprocess_robots(request: TableRequest, robots: Iterable[RobotModule]) -> tuple[_Candidate, ...]:
    available = [r for r in robots if r.available and r.top_flat]
    if not available:
        return ()

    target_height = request.target_height_m
    if target_height is None:
        target_height = median(r.height_m for r in available)

    out: list[_Candidate] = []
    for robot in sorted(available, key=lambda r: r.id):
        if abs(robot.height_m - target_height) > HEIGHT_TOLERANCE_M:
            continue
        out.append(_Candidate(robot, robot.length_m, robot.width_m, 0.0))
        if robot.can_rotate_90 and not isclose(robot.length_m, robot.width_m, abs_tol=EPS):
            out.append(_Candidate(robot, robot.width_m, robot.length_m, 1.5707963267948966))

    return tuple(out)


def _unique_candidate_robots(candidates: Iterable[_Candidate]) -> tuple[RobotModule, ...]:
    by_id: dict[str, RobotModule] = {}
    for candidate in candidates:
        by_id[candidate.robot.id] = candidate.robot
    return tuple(by_id.values())


def _area_tolerance(request: TableRequest) -> float:
    perimeter = 2.0 * (request.target_length_m + request.target_width_m)
    return max(request.tolerance_m * perimeter, request.tolerance_m * request.tolerance_m)


def _is_complete(metrics: dict[str, float], request: TableRequest) -> bool:
    return (
        metrics["uncovered_area"] <= _area_tolerance(request)
        and metrics["overhang_area"] <= request.max_overhang_m * (request.target_length_m + request.target_width_m)
    )


def _within_bounds(rect: _Rect, request: TableRequest) -> bool:
    return (
        rect.x >= -request.max_overhang_m - EPS
        and rect.y >= -request.max_overhang_m - EPS
        and rect.right <= request.target_length_m + request.max_overhang_m + EPS
        and rect.top <= request.target_width_m + request.max_overhang_m + EPS
    )


def _overlap(a: _Rect, b: _Rect) -> bool:
    return not (
        a.right <= b.x + EPS
        or b.right <= a.x + EPS
        or a.top <= b.y + EPS
        or b.top <= a.y + EPS
    )


def _first_uncovered_frontier_point(rects: tuple[_Rect, ...], request: TableRequest) -> tuple[float, float] | None:
    if not rects:
        return (0.0, 0.0)

    xs, ys = _compressed_axes(rects, request)
    for y0, y1 in zip(ys, ys[1:]):
        for x0, x1 in zip(xs, xs[1:]):
            if x1 - x0 <= EPS or y1 - y0 <= EPS:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if not _inside_target(cx, cy, request):
                continue
            if not any(_point_in_rect(cx, cy, rect) for rect in rects):
                if min(x1 - x0, y1 - y0) <= request.tolerance_m + EPS:
                    continue
                return (x0, y0)
    return None


def _compressed_axes(rects: tuple[_Rect, ...], request: TableRequest) -> tuple[list[float], list[float]]:
    xs = {0.0, request.target_length_m}
    ys = {0.0, request.target_width_m}
    for rect in rects:
        xs.update([_clamp_axis(rect.x, request.target_length_m), _clamp_axis(rect.right, request.target_length_m)])
        ys.update([_clamp_axis(rect.y, request.target_width_m), _clamp_axis(rect.top, request.target_width_m)])
    return sorted(xs), sorted(ys)


def _clamp_axis(value: float, upper: float) -> float:
    if abs(value) <= EPS:
        return 0.0
    if abs(value - upper) <= EPS:
        return upper
    return value


def _inside_target(x: float, y: float, request: TableRequest) -> bool:
    return -EPS <= x <= request.target_length_m + EPS and -EPS <= y <= request.target_width_m + EPS


def _point_in_rect(x: float, y: float, rect: _Rect) -> bool:
    return rect.x - EPS <= x <= rect.right + EPS and rect.y - EPS <= y <= rect.top + EPS


def _order_candidates_by_fit(
    candidates: tuple[_Candidate, ...],
    state: _State,
    anchor: tuple[float, float],
    request: TableRequest,
) -> tuple[_Candidate, ...]:
    def key(candidate: _Candidate) -> tuple[float, float, float, str]:
        rect = _Rect(candidate.robot.id, anchor[0], anchor[1], candidate.length, candidate.width, candidate.yaw_offset)
        protrusion = max(0.0, rect.right - request.target_length_m) + max(0.0, rect.top - request.target_width_m)
        distance = _distance_to_world_rect_center(candidate.robot, rect, request)
        return (protrusion, -candidate.length * candidate.width, distance, candidate.robot.id)

    return tuple(sorted(candidates, key=key))


def _measure_layout(rects: tuple[_Rect, ...], request: TableRequest) -> dict[str, float]:
    xs = {0.0, request.target_length_m}
    ys = {0.0, request.target_width_m}
    for rect in rects:
        xs.update([rect.x, rect.right])
        ys.update([rect.y, rect.top])

    x_sorted = sorted(xs)
    y_sorted = sorted(ys)
    uncovered = 0.0
    overhang = 0.0
    largest_gap = 0.0

    for x0, x1 in zip(x_sorted, x_sorted[1:]):
        for y0, y1 in zip(y_sorted, y_sorted[1:]):
            if x1 - x0 <= EPS or y1 - y0 <= EPS:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            area = (x1 - x0) * (y1 - y0)
            covered = any(_point_in_rect(cx, cy, rect) for rect in rects)
            inside = _inside_target(cx, cy, request)
            if inside and not covered:
                uncovered += area
                largest_gap = max(largest_gap, min(x1 - x0, y1 - y0))
            if covered and not inside:
                overhang += area

    return {
        "uncovered_area": uncovered,
        "overhang_area": overhang,
        "seam_length": _seam_length(rects),
        "max_gap": largest_gap,
    }


def _seam_length(rects: tuple[_Rect, ...]) -> float:
    seam = 0.0
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            if isclose(a.right, b.x, abs_tol=EPS) or isclose(b.right, a.x, abs_tol=EPS):
                seam += max(0.0, min(a.top, b.top) - max(a.y, b.y))
            if isclose(a.top, b.y, abs_tol=EPS) or isclose(b.top, a.y, abs_tol=EPS):
                seam += max(0.0, min(a.right, b.right) - max(a.x, b.x))
    return seam


def _score_metrics(metrics: dict[str, float], placement_count: int, movement_distance: float) -> float:
    return (
        10_000.0 * metrics["uncovered_area"]
        + 5_000.0 * metrics["overhang_area"]
        + 500.0 * metrics["max_gap"]
        + 50.0 * placement_count
        + 10.0 * metrics["seam_length"]
        + movement_distance
    )


def _total_distance(rects: tuple[_Rect, ...], candidates: tuple[_Candidate, ...], request: TableRequest) -> float:
    by_id = {candidate.robot.id: candidate.robot for candidate in candidates}
    return sum(_distance_to_world_rect_center(by_id[rect.robot_id], rect, request) for rect in rects)


def _distance_to_world_rect_center(robot: RobotModule, rect: _Rect, request: TableRequest) -> float:
    x, y = _local_center_to_world(rect.center, request)
    return hypot(robot.current_pose.x_m - x, robot.current_pose.y_m - y)


def _local_center_to_world(local_center: tuple[float, float], request: TableRequest) -> tuple[float, float]:
    dx = local_center[0] - request.target_length_m / 2.0
    dy = local_center[1] - request.target_width_m / 2.0
    yaw = request.target_yaw
    return (
        request.target_pose.x_m + cos(yaw) * dx - sin(yaw) * dy,
        request.target_pose.y_m + sin(yaw) * dx + cos(yaw) * dy,
    )


def _make_plan(
    rects: tuple[_Rect, ...],
    candidates: tuple[_Candidate, ...],
    request: TableRequest,
    status: PlanStatus,
) -> CompositionPlan:
    metrics = _measure_layout(rects, request)
    placements = tuple(_make_placement(rect, request) for rect in sorted(rects, key=lambda r: (r.y, r.x, r.robot_id)))
    score = _score_metrics(metrics, len(placements), _total_distance(rects, candidates, request))
    return CompositionPlan(
        placements=placements,
        score=score,
        uncovered_area=metrics["uncovered_area"],
        overhang_area=metrics["overhang_area"],
        seam_length=metrics["seam_length"],
        max_gap=metrics["max_gap"],
        status=status,
    )


def _make_placement(rect: _Rect, request: TableRequest) -> Placement:
    world_x, world_y = _local_center_to_world(rect.center, request)
    return Placement(
        robot_id=rect.robot_id,
        center_x=world_x,
        center_y=world_y,
        yaw=request.target_yaw + rect.yaw_offset,
        rect_length=rect.length,
        rect_width=rect.width,
    )


def _plan_sort_key(plan: CompositionPlan) -> tuple[float, int, tuple[str, ...]]:
    return (round(plan.score, 9), len(plan.placements), tuple(p.robot_id for p in plan.placements))


def _state_signature(state: _State) -> tuple[tuple[str, float, float, float, float], ...]:
    return tuple(
        sorted(
            (
                rect.robot_id,
                round(rect.x, 6),
                round(rect.y, 6),
                round(rect.length, 6),
                round(rect.width, 6),
            )
            for rect in state.rects
        )
    )


def _largest_gap_dimension(metrics: dict[str, float]) -> float:
    return metrics["max_gap"]


def _plan_approximate_cover(
    best_partial: tuple[float, _State] | None,
    candidates: tuple[_Candidate, ...],
    request: TableRequest,
) -> CompositionPlan | None:
    if best_partial is None:
        return None

    _, state = best_partial
    metrics = _measure_layout(state.rects, request)
    target_area = request.target_length_m * request.target_width_m
    if metrics["uncovered_area"] <= max(_area_tolerance(request) * 4.0, target_area * 0.03):
        return _make_plan(state.rects, candidates, request, PlanStatus.APPROXIMATE)
    return None


__all__ = [
    "CompositionPlan",
    "Placement",
    "PlanStatus",
    "Pose",
    "RobotModule",
    "TableRequest",
    "plan_table_composition",
]
