# src/core/flow_solver.py
from ortools.linear_solver import pywraplp
import time
import logging

# Настройка логгера (если еще не настроен)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class FlowSolver:
    def __init__(self, model):
        self.model = model
        self.solver = model.get_solver()
        logging.debug("FlowSolver initialized.")

    def solve(self):
        logging.info("Starting solving process using FlowModel...")
        start_time = time.time()
        status = self.solver.Solve()
        end_time = time.time()
        logging.info(f"Solving took {end_time - start_time:.3f} seconds.")

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            logging.info("Solution found. Extracting the solution...")
            solution = self.get_solution()
            logging.debug(f"Solution: {solution}")
            if not self.is_solution_valid(solution):
                logging.warning("The solution is invalid: Bridge numbers does not match.")
                return pywraplp.Solver.INFEASIBLE, None
            return status, solution
        else:
            logging.warning(f"No solution found. Solver status: {status}")
            return status, None

    def get_solution(self):
        x, y = self.model.get_variables()
        solution = {}
        for (i, j), var in x.items():
            bridge_count = 0
            if var.SolutionValue() > 0.5:
                bridge_count += 1
            if y[i, j].SolutionValue() > 0.5:
                bridge_count += 1
            if bridge_count > 0:
                solution[i, j] = bridge_count
        return solution

    def is_solution_valid(self, solution):
        islands = self.model.puzzle.get_all_islands()
        for island in islands:
            island_id = island['id']
            degree = island['degree']

            # Считаем количество мостов, прилегающих к острову
            bridges_count = 0
            for (i, j), bridges in solution.items():
                if i == island_id or j == island_id:
                    bridges_count += bridges

            if bridges_count != degree:
                logging.warning(
                    f"Island {island_id} has incorrect bridge count: expected {degree}, got {bridges_count}")
                return False  # Если количество мостов не совпадает с требуемой степенью, решение неверное

        return True  # Если все острова имеют правильное количество мостов, решение верное
