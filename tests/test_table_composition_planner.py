import unittest

from src.table_composition_planner import (
    PlanStatus,
    Pose,
    RobotModule,
    TableRequest,
    plan_table_composition,
)


def robot(robot_id, length=1.0, width=1.0, x=0.0, y=0.0, height=0.72):
    return RobotModule(
        id=robot_id,
        length_m=length,
        width_m=width,
        height_m=height,
        current_pose=Pose(x, y, 0.0),
        max_load=20.0,
    )


class TableCompositionPlannerTests(unittest.TestCase):
    def test_exact_fit_with_six_square_modules(self):
        request = TableRequest(target_length_m=3.0, target_width_m=2.0, target_height_m=0.72)
        robots = [robot(f"r{i}") for i in range(6)]

        plan = plan_table_composition(request, robots)

        self.assertEqual(plan.status, PlanStatus.FEASIBLE)
        self.assertEqual(len(plan.placements), 6)
        self.assertAlmostEqual(plan.uncovered_area, 0.0)

    def test_mixed_fit_with_rotated_rectangles(self):
        request = TableRequest(target_length_m=3.0, target_width_m=2.0, target_height_m=0.72)
        robots = [
            robot("a", length=2.0, width=1.0),
            robot("b", length=1.0, width=2.0),
            robot("c", length=2.0, width=1.0),
        ]

        plan = plan_table_composition(request, robots)

        self.assertEqual(plan.status, PlanStatus.FEASIBLE)
        self.assertEqual(len(plan.placements), 3)
        self.assertAlmostEqual(plan.uncovered_area, 0.0)

    def test_duplicate_robots_choose_lower_movement_distance(self):
        request = TableRequest(target_length_m=2.0, target_width_m=1.0, target_height_m=0.72)
        robots = [
            robot("far", x=100.0, y=100.0),
            robot("near_left", x=-0.5, y=0.0),
            robot("near_right", x=0.5, y=0.0),
        ]

        plan = plan_table_composition(request, robots)
        used = {p.robot_id for p in plan.placements}

        self.assertEqual(plan.status, PlanStatus.FEASIBLE)
        self.assertEqual(used, {"near_left", "near_right"})

    def test_tolerance_fit_accepts_small_uncovered_strip(self):
        request = TableRequest(
            target_length_m=3.01,
            target_width_m=2.0,
            target_height_m=0.72,
            tolerance_m=0.02,
        )
        robots = [robot(f"r{i}") for i in range(6)]

        plan = plan_table_composition(request, robots)

        self.assertEqual(plan.status, PlanStatus.FEASIBLE)
        self.assertLessEqual(plan.uncovered_area, 0.05)

    def test_infeasible_when_area_is_insufficient(self):
        request = TableRequest(target_length_m=3.0, target_width_m=2.0, target_height_m=0.72)
        robots = [robot("a"), robot("b")]

        plan = plan_table_composition(request, robots)

        self.assertEqual(plan.status, PlanStatus.NO_SOLUTION)

    def test_infeasible_when_height_does_not_match(self):
        request = TableRequest(target_length_m=2.0, target_width_m=1.0, target_height_m=0.72)
        robots = [robot("a", height=0.9), robot("b", height=0.9)]

        plan = plan_table_composition(request, robots)

        self.assertEqual(plan.status, PlanStatus.NO_SOLUTION)

    def test_deterministic_output_for_same_input(self):
        request = TableRequest(target_length_m=3.0, target_width_m=2.0, target_height_m=0.72)
        robots = [robot(f"r{i}") for i in range(6)]

        first = plan_table_composition(request, robots)
        second = plan_table_composition(request, robots)

        self.assertEqual(first.status, second.status)
        self.assertEqual(first.placements, second.placements)


if __name__ == "__main__":
    unittest.main()

