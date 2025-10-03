# src/core/solver.py
import time
import networkx as nx
from ortools.linear_solver import pywraplp
import logging

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(message)s')

class Solver:
    def __init__(self, model):
        self.model = model
        self.solver = model.get_solver()
        self.single_bridge, self.double_bridge = model.get_variables()
        self.max_iterations = 200

    def solve(self):
        """Решает головоломку и возвращает решение."""
        logging.info("Starting solving process...")
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            logging.info("Solution found. Extracting the solution...")
            solution = self.get_solution()
            return status, solution
        else:
            logging.warning(f"No solution found. Solver status: {status}")
            return status, None

    def get_solution(self):
        """Извлекает решение из солвера."""
        x, y = self.model.get_variables()
        solution = {}
        for (i, j), var in x.items():
            bridge_count = 0
            if var.SolutionValue() > 0.5:  # Порог для одинарного моста
                bridge_count += 1
            if y[i, j].SolutionValue() > 0.5:  # Порог для двойного моста
                bridge_count += 1
            if bridge_count > 0:
                solution[i, j] = bridge_count
        return solution

    def find_connected_components(self, solution):
        """Находит связные компоненты в текущем решении."""
        graph = nx.Graph()
        islands = self.model.puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'])
        if solution:
            for (i, j), bridges in solution.items():
                graph.add_edge(i, j)

        return list(nx.connected_components(graph))

    def is_solution_valid(self, solution):
        """Проверяет, что решение удовлетворяет требованиям по количеству мостов для каждого острова."""
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
                logging.warning(f"Island {island_id} has incorrect bridge count: expected {degree}, got {bridges_count}")
                return False  # Если количество мостов не совпадает с требуемой степенью, решение неверное

        return True  # Если все острова имеют правильное количество мостов, решение верное

    def solve_with_cuts(self):
        """Решает головоломку с итеративным добавлением ограничений разрезов. Возвращает (status, solution, iteration_count)."""
        start_time = time.time()
        iteration_count = 0
        for i in range(self.max_iterations):
            iteration_count += 1
            logging.info(f"Starting iteration {iteration_count}...")
            status, solution = self.solve()
            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                if not self.is_solution_valid(solution):
                    logging.warning("Solution is invalid: Incorrect bridge count for some islands.")
                    continue
                components = self.find_connected_components(solution)
                if len(components) == 1:
                    logging.info(f"Solution found with 1 connected component in {iteration_count} iterations.")
                    end_time = time.time()
                    logging.info(f"Problem solved in {end_time - start_time:.3f} seconds")
                    return status, solution, iteration_count
                components.sort(key=len, reverse=True)
                for component in components[1:]:
                    self.model.add_cut(component)
                    logging.info(f"Cut added for component: {component}")
            else:
                logging.warning("No solution found in the current iteration.")
                logging.info("Maximum iterations reached. No solution found.")
                return status, None, iteration_count
        logging.warning("Maximum iterations reached. No solution found.")
        return status, None, iteration_count