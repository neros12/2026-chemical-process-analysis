from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


class TridiagonalMatrix:
    def __init__(self, row_count: int, column_count: int) -> None:
        self.values = [[0.0 for _ in range(column_count)] for _ in range(row_count)]

    def __getitem__(self, index: tuple[int, int]) -> float:
        row_index, column_index = index
        return self.values[row_index][column_index]

    def __setitem__(self, index: tuple[int, int], value: float) -> None:
        row_index, column_index = index
        self.values[row_index][column_index] = value


LOWER = 1
DIAGONAL = 2
UPPER = 3
DEFAULT_DRAW_RATIO = 10.0
DEFAULT_DISTURBANCE = 1.01
DEFAULT_DIMENSIONLESS_TIME = 0.01


@dataclass
class TransientNewtonianFluidSolver:
    nx: int = 500
    tol: float = 1e-10
    max_iter: int = 10_000
    area: list[float] = field(init=False)
    old_area: list[float] = field(init=False)
    velocity: list[float] = field(init=False)
    old_velocity: list[float] = field(init=False)
    temp_velocity: list[float] = field(init=False)
    matrix: TridiagonalMatrix = field(init=False)
    rhs: list[float] = field(init=False)
    dx: float = field(init=False)
    dt: float = field(init=False)
    coeff: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = [0.0 for _ in range(self.nx + 1)]
        self.old_area = [0.0 for _ in range(self.nx + 1)]
        self.velocity = [0.0 for _ in range(self.nx + 1)]
        self.old_velocity = [0.0 for _ in range(self.nx + 1)]
        self.temp_velocity = [0.0 for _ in range(self.nx + 1)]
        self.matrix = TridiagonalMatrix(self.nx + 1, 4)
        self.rhs = [0.0 for _ in range(self.nx + 1)]

        self.dx = 1.0 / float(self.nx)
        self.dt = self.dx / 5.0
        self.coeff = 8.0 * self.dx / self.dt

    def initialize_steady_state(self, draw_ratio: float) -> None:
        self.velocity[self.nx] = draw_ratio
        for grid_index in range(self.nx + 1):
            self.velocity[grid_index] = draw_ratio ** (self.dx * float(grid_index))
            self.area[grid_index] = 1.0 / self.velocity[grid_index]

    def solve_area_matrix(self) -> None:
        for grid_index in range(1, self.nx):
            self.matrix[grid_index, LOWER] = (
                -self.old_velocity[grid_index - 1] - self.velocity[grid_index - 1]
            )
            self.matrix[grid_index, DIAGONAL] = self.coeff
            self.matrix[grid_index, UPPER] = (
                self.old_velocity[grid_index + 1] + self.velocity[grid_index + 1]
            )
            self.rhs[grid_index] = (
                (self.old_velocity[grid_index - 1] + self.velocity[grid_index - 1])
                * self.old_area[grid_index - 1]
                + self.coeff * self.old_area[grid_index]
                - (self.old_velocity[grid_index + 1] + self.velocity[grid_index + 1])
                * self.old_area[grid_index + 1]
            )

        self.rhs[1] = (
            (self.old_velocity[0] + self.velocity[0]) * self.old_area[0]
            + self.coeff * self.old_area[1]
            - (self.old_velocity[2] + self.velocity[2]) * self.old_area[2]
            + (self.old_velocity[0] + self.velocity[0])
        )
        self.matrix[1, LOWER] = 0.0

        self.matrix[self.nx, LOWER] = (
            -4.0 * self.old_velocity[self.nx - 1]
            - 4.0 * self.velocity[self.nx - 1]
            + self.coeff
        )
        self.matrix[self.nx, DIAGONAL] = (
            4.0 * self.old_velocity[self.nx] + 4.0 * self.velocity[self.nx] + self.coeff
        )
        self.matrix[self.nx, UPPER] = 0.0
        self.rhs[self.nx] = (
            self.coeff
            + 4.0 * self.old_velocity[self.nx - 1]
            + 4.0 * self.velocity[self.nx - 1]
        ) * self.old_area[self.nx - 1] + (
            self.coeff - 4.0 * self.old_velocity[self.nx] - 4.0 * self.velocity[self.nx]
        ) * self.old_area[
            self.nx
        ]

        self.matrix[self.nx, LOWER] /= self.matrix[self.nx, DIAGONAL]
        self.rhs[self.nx] /= self.matrix[self.nx, DIAGONAL]
        self.solve_diagonal(self.area, self.nx)

    def solve_velocity_matrix(self) -> None:
        for grid_index in range(1, self.nx):
            self.matrix[grid_index, LOWER] = (
                self.area[grid_index - 1]
                + 4.0 * self.area[grid_index]
                - self.area[grid_index + 1]
            )
            self.matrix[grid_index, DIAGONAL] = -8.0 * self.area[grid_index]
            self.matrix[grid_index, UPPER] = (
                -self.area[grid_index - 1]
                + 4.0 * self.area[grid_index]
                + self.area[grid_index + 1]
            )
            self.rhs[grid_index] = 0.0

        self.matrix[1, LOWER] = 0.0
        self.matrix[self.nx - 1, UPPER] = 0.0
        self.rhs[1] = -self.area[0] - 4.0 * self.area[1] + self.area[2]
        self.rhs[self.nx - 1] = (
            self.area[self.nx - 2] - 4.0 * self.area[self.nx - 1] - self.area[self.nx]
        ) * self.velocity[self.nx]

        self.matrix[self.nx - 1, LOWER] /= self.matrix[self.nx - 1, DIAGONAL]
        self.rhs[self.nx - 1] /= self.matrix[self.nx - 1, DIAGONAL]
        self.solve_diagonal(self.velocity, self.nx - 1)

    def solve_diagonal(self, solution: list[float], size: int) -> None:
        for step_index in range(2, size + 1):
            matrix_index = size - step_index + 2
            denominator = (
                self.matrix[matrix_index - 1, DIAGONAL]
                - self.matrix[matrix_index, LOWER]
                * self.matrix[matrix_index - 1, UPPER]
            )
            factor = 1.0 / denominator
            self.matrix[matrix_index - 1, LOWER] *= factor
            self.rhs[matrix_index - 1] = (
                self.rhs[matrix_index - 1]
                - self.matrix[matrix_index - 1, UPPER] * self.rhs[matrix_index]
            ) * factor

        solution[1] = self.rhs[1]
        for grid_index in range(2, size + 1):
            solution[grid_index] = (
                self.rhs[grid_index]
                - self.matrix[grid_index, LOWER] * solution[grid_index - 1]
            )

    def solve_time_step(self, new_draw_ratio: float) -> int:
        self.velocity[self.nx] = new_draw_ratio

        for iteration in range(1, self.max_iter + 1):
            self.temp_velocity[:] = self.velocity
            self.solve_area_matrix()
            self.solve_velocity_matrix()

            max_difference = max(
                abs(new_velocity - old_velocity)
                for new_velocity, old_velocity in zip(self.velocity, self.temp_velocity)
            )
            if max_difference <= self.tol:
                return iteration

        raise RuntimeError(
            f"ERROR:: 최대 iteration({self.max_iter}) 안에 수렴하지 못했습니다!"
        )

    def simulate(
        self,
        draw_ratio: float,
        disturbance: float,
        dimensionless_time: float,
        *,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        output_path = (
            Path(output_dir)
            if output_dir is not None
            else Path(__file__).resolve().parent
        )
        output_path.mkdir(parents=True, exist_ok=True)

        self.initialize_steady_state(draw_ratio)
        new_draw_ratio = draw_ratio * disturbance
        print(f"new draw ratio: {new_draw_ratio}")

        self.write_result(output_path / "18a.dat", 0.0, self.area[self.nx])
        print(0, 0.0, self.area[self.nx])

        total_steps = int(dimensionless_time / self.dt)
        print(f"ntime: {total_steps}")

        for time_step_index in range(1, total_steps + 1):
            time_step = self.dt * float(time_step_index)
            self.old_velocity[:] = self.velocity
            self.old_area[:] = self.area

            self.solve_time_step(new_draw_ratio)

            if time_step <= 10.0:
                filename = "18a.dat"
            elif time_step <= 20.0:
                filename = "18b.dat"
            elif time_step <= 30.0:
                filename = "18c.dat"
            else:
                continue

            if time_step_index % 100 == 0:
                print(time_step_index, time_step, self.area[self.nx])
            if time_step_index % 5 == 0:
                self.write_result(output_path / filename, time_step, self.area[self.nx])

    @staticmethod
    def write_result(file_path: Path, time_step: float, area_at_exit: float) -> None:
        with file_path.open("a", encoding="utf-8") as file:
            file.write(f"{time_step:.16e} {area_at_exit:.16e}\n")


def read_float_values(prompt: str, count: int) -> list[float]:
    print(prompt)
    values: list[float] = []
    while len(values) < count:
        try:
            line = input()
        except EOFError:
            return []
        values.extend(float(value) for value in line.split())
    return values[:count]


def read_simulation_inputs() -> tuple[float, ...]:
    if len(sys.argv) > 1:
        if len(sys.argv) != 4:
            raise SystemExit(
                "Usage: python 01.py <draw_ratio> <disturbance> <dimensionless_time>"
            )

        return tuple(float(value) for value in sys.argv[1:4])

    draw_ratio_and_disturbance = read_float_values(
        "draw ratio and disturbance :",
        2,
    )
    if len(draw_ratio_and_disturbance) == 2:
        dimensionless_time = read_float_values("dimensionless time :", 1)
        if len(dimensionless_time) == 1:
            return (
                draw_ratio_and_disturbance[0],
                draw_ratio_and_disturbance[1],
                dimensionless_time[0],
            )

    print("입력을 읽을 수 없어 기본값으로 실행합니다.")
    print(
        "draw ratio = "
        f"{DEFAULT_DRAW_RATIO}, disturbance = {DEFAULT_DISTURBANCE}, "
        f"dimensionless time = {DEFAULT_DIMENSIONLESS_TIME}"
    )
    return DEFAULT_DRAW_RATIO, DEFAULT_DISTURBANCE, DEFAULT_DIMENSIONLESS_TIME


if __name__ == "__main__":
    draw_ratio_input, disturbance_input, dimensionless_time_input = (
        read_simulation_inputs()
    )

    solver = TransientNewtonianFluidSolver()
    solver.simulate(draw_ratio_input, disturbance_input, dimensionless_time_input)
