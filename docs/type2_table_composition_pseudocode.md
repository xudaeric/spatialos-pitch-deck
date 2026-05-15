# 第二类机器人桌面组合 Planner 伪代码

## 1. 目标

给定用户请求，例如“我需要一个 N 米 x M 米的桌面”（演示版随机取 1-10 米内整数），系统从一组长方形柜式第二类机器人中选择最合适组合，并输出每个机器人的最终：

- `position`：目标中心坐标；
- `rectangle`：占用的桌面矩形；
- `direction`：最终朝向。

本版只解决“最终排列”问题，不输出完整机器人路径。移动路径、空间预约、避障、互锁和交通调度应进入下一层 motion planner。展示动画采用 parallel reserved lanes：所有被选中的柜式机器人从房间不同位置同时移动；每个机器人沿自己的预约通道向最终 footprint 收敛，因此移动中不重叠、不碰撞；真实硬件仍需由 Nav2/BT/reservation table 做路径验证。

## 2. Rush Hour 论文对本问题的启发

Flake 和 Baum 的 Rush Hour PSPACE-complete 证明说明：多个固定朝向长方形块在受限空间内移动时，完整可达性搜索会变得非常复杂。它构造了 `CROSSOVER`、`BOTH`、`EITHER` 等 primitive devices，让简单长方形移动也能表达复杂逻辑。

因此这里采用分层设计：

- `table composition`：只求最终桌面覆盖方案；
- `motion planning`：后续再判断机器人是否能安全到达这些最终 pose；
- `reservation/safety`：未来借鉴 Rush Hour 的 primitive constraints，用空间锁表达 `CROSSOVER`，用多条件门表达 `BOTH`，用备选方案表达 `EITHER`。

这能避免实时桌面组合器被完整运动规划拖入 PSPACE-hard 的状态空间。

## 3. 数据接口

```pseudo
type Pose:
    x_m
    y_m
    yaw_rad

type RobotModule:
    id
    length_m
    width_m
    height_m
    current_pose
    top_flat = true
    available = true
    max_load
    can_rotate_90 = true

type TableRequest:
    target_length_m
    target_width_m
    target_pose
    target_yaw
    target_height_m = optional
    tolerance_m = 0.02
    max_gap_m = 0.02
    max_overhang_m = 0.03
    time_budget_ms = 500

type Placement:
    robot_id
    center_x
    center_y
    yaw
    rect_length
    rect_width

type CompositionPlan:
    placements[]
    score
    uncovered_area
    overhang_area
    seam_length
    status  // FEASIBLE, APPROXIMATE, NO_SOLUTION
```

## 4. 主流程

```pseudo
function PLAN_TABLE_COMPOSITION(request, robots):
    candidates = PREPROCESS_ROBOTS(request, robots)
    if SUM_TOP_AREA(candidates) < TARGET_AREA(request) - AREA_TOLERANCE(request):
        return NO_SOLUTION

    best_plan = null
    best_partial = EMPTY_LAYOUT(request)
    deadline = now() + request.time_budget_ms

    queue = PRIORITY_QUEUE()
    PUSH(queue, EMPTY_LAYOUT(request), priority = 0)

    while queue is not empty and now() < deadline:
        partial = POP_MOST_PROMISING_STATE(queue)

        metrics = MEASURE_COVERAGE(partial, request)
        best_partial = MIN_BY_SCORE(best_partial, partial)

        if IS_COMPLETE(metrics, request):
            assigned = ASSIGN_REAL_ROBOTS(partial, candidates, request)
            scored = SCORE_PLAN(assigned, request)
            best_plan = MIN_BY_SCORE(best_plan, scored)
            continue

        anchor = FIRST_UNCOVERED_FRONTIER_POINT(partial, request)

        for candidate in ORDER_CANDIDATES_BY_FIT(candidates, partial, anchor, request):
            if candidate.robot_id in partial.used_robot_ids:
                continue

            for orientation in LEGAL_ORIENTATIONS(candidate):
                rect = PLACE_RECT_AT_ANCHOR(candidate, orientation, anchor)

                if not WITHIN_BOUNDS(rect, request, request.max_overhang_m):
                    continue
                if OVERLAPS_EXISTING(rect, partial):
                    continue
                if CREATES_INVALID_GAP(rect, partial, request.max_gap_m):
                    continue

                next_state = ADD_RECT(partial, rect)
                PUSH(queue, next_state, HEURISTIC(next_state, request))

    if best_plan != null:
        return best_plan

    approximate = PLAN_APPROXIMATE_COVER(best_partial, candidates, request)
    if approximate != null:
        return approximate

    return NO_SOLUTION
```

## 5. 关键子程序

```pseudo
function PREPROCESS_ROBOTS(request, robots):
    target_height = request.target_height_m or MEDIAN_AVAILABLE_HEIGHT(robots)
    out = []

    for r in robots:
        if not r.available or not r.top_flat:
            continue
        if ABS(r.height_m - target_height) > HEIGHT_TOLERANCE:
            continue

        out.append((r.id, r.length_m, r.width_m, yaw_offset = 0))

        if r.can_rotate_90 and r.length_m != r.width_m:
            out.append((r.id, r.width_m, r.length_m, yaw_offset = 90deg))

    return out
```

```pseudo
function FIRST_UNCOVERED_FRONTIER_POINT(partial, request):
    cells = COMPRESS_RECTANGLE_EDGES(partial.rectangles, request.target_bounds)

    for cell in cells sorted by y then x:
        if cell.center is inside target_bounds and not COVERED(cell.center, partial):
            return cell.lower_left

    return null
```

```pseudo
function SCORE_PLAN(plan, request):
    return
        10000 * plan.uncovered_area +
        5000  * plan.overhang_area +
        500   * plan.max_gap +
        50    * COUNT(plan.placements) +
        10    * plan.seam_length +
        1     * TOTAL_DISTANCE_FROM_CURRENT_POSES(plan)
```

```pseudo
function ASSIGN_REAL_ROBOTS(shape_layout, candidates, request):
    assignments = []
    unused = GROUP_ROBOTS_BY_FOOTPRINT(candidates)

    for rect in shape_layout.rectangles sorted by position:
        robots = unused[rect.length, rect.width]
        chosen = ARGMIN(robots, DISTANCE(robot.current_pose, rect.center))
        assignments.add(MAKE_PLACEMENT(chosen, rect, request.target_pose, request.target_yaw))
        REMOVE_FROM_POOL(chosen)

    return assignments
```

## 6. 状态合法性

每个 placement 必须满足：

- 不与已有机器人矩形重叠；
- 总覆盖区域足以覆盖目标桌面；
- 未覆盖区域不超过容差；
- 外溢不超过 `max_overhang_m`；
- 缝隙不超过 `max_gap_m`；
- 高度在容差内；
- 同一实体机器人只能被使用一次；
- 若后续 motion planner 判定无法到达，则返回上层重新选择组合。

## 7. 默认参数

```pseudo
tolerance_m = 0.02
max_gap_m = 0.02
max_overhang_m = 0.03
height_tolerance_m = 0.03
time_budget_ms = 500
```

## 8. 验收场景

- Exact fit：`1m x 1m` 模块组成任意 `1-10m` 整数长宽矩形，最终无缝覆盖。
- Mixed fit：`2m x 1m`、`1m x 2m`、`1m x 1m` 混合覆盖。
- Duplicate robots：相同尺寸机器人中选择移动距离更小的实体。
- Tolerance fit：目标边长比模块总边长多 1cm，仍可接受。
- Infeasible：面积不足或高度不一致时返回 `NO_SOLUTION`。
- Determinism：同样输入多次运行，按 `score -> movement -> robot_id` 稳定输出。
