# src/core/model_flow.py
from ortools.linear_solver import pywraplp

class FlowModel:
    def __init__(self, puzzle, target_node=None):  # Добавляем target_node
        self.puzzle = puzzle
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.x = {}
        self.y = {}
        self.f = {}  # Потоковые переменные
        self.all_islands = [island['id'] for island in puzzle.get_all_islands()]
        self.num_islands = len(self.all_islands)

        # Выбор целевого узла. Если не указан, выбираем первый попавшийся
        if target_node is None:
            self.target_node = self.all_islands[0]
        else:
            if target_node not in self.all_islands:
                raise ValueError("Target node must be one of the islands in the puzzle.")
            self.target_node = target_node

        # Для хранения ограничений
        self.degree_constraints = []
        self.intersection_constraints = []
        self.connectivity_constraint = None

        self.build_variables()
        self.build_constraints()

    def get_solver(self):
        return self.solver

    def get_variables(self):
        return self.x, self.y

    def build_variables(self):
        """Создаёт переменные для модели."""
        self.x = {}
        self.y = {}
        self.f = {}

        for island1_id, island2_id in self.puzzle.get_all_edges():
            self.x[island1_id, island2_id] = self.solver.IntVar(0, 1, f'x_{island1_id}_{island2_id}')
            self.y[island1_id, island2_id] = self.solver.IntVar(0, 1, f'y_{island1_id}_{island2_id}')

            # Создаем потоковые переменные для обоих направлений
            self.f[island1_id, island2_id] = self.solver.NumVar(0, self.num_islands - 1,
                                                                  f'f_{island1_id}_{island2_id}')
            self.f[island2_id, island1_id] = self.solver.NumVar(0, self.num_islands - 1,
                                                                  f'f_{island2_id}_{island1_id}')

    def build_constraints(self):
        """Создаёт ограничения для модели."""

        # (1) Ограничение на степень вершины (уравнение 1 в документе)
        for island in self.puzzle.get_all_islands():
            island_id = island['id']
            degree = island['degree']
            neighbors = self.puzzle.get_adjacency()[island_id]
            constraint = self.solver.Add(sum(self.x[min(island_id, neighbor_id), max(island_id, neighbor_id)] +
                               self.y[min(island_id, neighbor_id), max(island_id, neighbor_id)]
                               for neighbor_id in neighbors) == degree)
            self.degree_constraints.append(constraint)

        # (2) Ограничение кратных рёбер (уравнение 2 в документе)
        for island1_id, island2_id in self.puzzle.get_all_edges():
            constraint1 = self.solver.Add(self.y[island1_id, island2_id] <= self.x[island1_id, island2_id])
            self.degree_constraints.append(constraint1)

        # (3) Ограничение пересечений (уравнение 3 в документе)
        for intersection in self.puzzle.get_intersections():
            (i, j), (k, l) = list(intersection)
            constraint = self.solver.Add(self.x[min(i, j), max(i, j)] + self.x[min(k, l), max(k, l)] <= 1)
            self.intersection_constraints.append(constraint)

        # (4) Ограничение слабой связности
        constraint = self.solver.Add(sum(self.x[island1_id, island2_id] for island1_id, island2_id in self.x) >= self.num_islands - 1)
        self.connectivity_constraint = constraint

        # Ограничения для потока

        self.solver.Add(sum(self.f[i, self.target_node] for i in self.all_islands if
                             (i, self.target_node) in self.f) == self.num_islands - 1)

        for i in self.all_islands:
            if i != self.target_node:
                self.solver.Add(sum(self.f[i, j] for j in self.all_islands if (i, j) in self.f) ==
                                sum(self.f[j, i] for j in self.all_islands if (j, i) in self.f) + 1)

        for island1_id, island2_id in self.puzzle.get_all_edges():
            self.solver.Add(self.f[island1_id, island2_id] + self.f[island2_id, island1_id] <=
                            (self.num_islands - 1) * self.x[island1_id, island2_id])