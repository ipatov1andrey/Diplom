from ortools.linear_solver import pywraplp


class Model:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.single_bridge = {}  # x_ij - переменная, равная 1, если есть ОДИН мост между i и j
        self.double_bridge = {}  # y_ij - переменная, равная 1, если есть ДВА моста между i и j
        self.all_islands = [island['id'] for island in self.puzzle.get_all_islands()]

        # Для хранения ограничений с выражениями (только для отладки)
        self.degree_constraints = []  # Ограничения степени вершины
        self.intersection_constraints = []  # Ограничения пересечений
        self.connectivity_constraint = None  # Ограничение слабой связности
        self.cut_constraints = []  # Ограничения разрезов

        self._build_variables()
        self._build_constraints()

    def get_solver(self):
        return self.solver

    def get_variables(self):
        return self.single_bridge, self.double_bridge

    def _build_variables(self):
        """Создаёт переменные для модели."""
        for island1_id, island2_id in self.puzzle.get_all_edges():
            self.single_bridge[island1_id, island2_id] = self.solver.IntVar(0, 1, f'single_{island1_id}_{island2_id}')
            self.double_bridge[island1_id, island2_id] = self.solver.IntVar(0, 1, f'double_{island1_id}_{island2_id}')

    def _build_constraints(self):
        """Создаёт ограничения для модели."""

        # (1) Ограничение на степень вершины (уравнение 1 в документе)
        for island in self.puzzle.get_all_islands():
            island_id = island['id']
            degree = island['degree']
            neighbors = self.puzzle.get_adjacency()[island_id]

            constraint = self.solver.Add(
                sum(self.single_bridge[min(island_id, neighbor_id), max(island_id, neighbor_id)] +
                    self.double_bridge[min(island_id, neighbor_id), max(island_id, neighbor_id)]
                    for neighbor_id in neighbors) == degree)
            self.degree_constraints.append(constraint)

        # (2) Ограничение кратных рёбер (уравнение 2 в документе)
        for island1_id, island2_id in self.puzzle.get_all_edges():
            constraint = self.solver.Add(
                self.double_bridge[island1_id, island2_id] <= self.single_bridge[island1_id, island2_id])
            self.degree_constraints.append(constraint)

        # (3) Ограничение пересечений (уравнение 3 в документе)
        for intersection in self.puzzle.get_intersections():
            (i, j), (k, l) = list(intersection)
            i1, j1 = min(i, j), max(i, j)
            k1, l1 = min(k, l), max(k, l)
            # Важное исправление: используем только x (single), т.к. x=1 означает наличие хотя бы одного моста.
            # Это разрешает двойной мост на одном ребре, но запрещает одновременную активность пересекающихся рёбер.
            constraint = self.solver.Add(
                self.single_bridge[i1, j1] + self.single_bridge[k1, l1] <= 1)
            self.intersection_constraints.append(constraint)

        # (4) Ограничение слабой связности (уравнение 4 в документе)
        num_islands = len(self.puzzle.get_all_islands())
        constraint = self.solver.Add(sum(self.single_bridge[island1_id, island2_id] for island1_id, island2_id in
                                         self.single_bridge) >= num_islands - 1)
        self.connectivity_constraint = constraint

    def add_cut(self, component):
        """Добавляет ограничение разреза для заданного подмножества островов."""
        all_islands = [island['id'] for island in self.puzzle.get_all_islands()]
        cut_complement = [island_id for island_id in all_islands if island_id not in component]
        constraint = self.solver.Add(sum(self.single_bridge[min(i, j), max(i, j)]
                                         for i in component for j in cut_complement
                                         if (i, j) in self.single_bridge or (j, i) in self.single_bridge) >= 1)
        self.cut_constraints.append(constraint)

    def print_variables_and_constraints(self):
        """Выводит переменные и ограничения модели."""
        print("Переменные:")
        for (i, j), var in self.single_bridge.items():
            print(f"x_{i},{j} ∈ {{0, 1}}")
        for (i, j), var in self.double_bridge.items():
            print(f"y_{i},{j} ∈ {{0, 1}}")

        print("\nОграничения степени вершины:")
        for constraint, expression in self.degree_constraints:
            print(expression)

        print("\nОграничения кратных рёбер:")
        # for constraint, expression in self.degree_constraints:
        #     print(expression)
        # print("\nОграничения кратных рёбер:")
        # for constraint, expression in self.degree_constraints:
        #   print(expression)

        print("\nОграничения пересечений:")
        for constraint, expression in self.intersection_constraints:
            print(expression)

        print("\nОграничение слабой связности:")
        if self.connectivity_constraint:
            constraint, expression = self.connectivity_constraint
            print(expression)

        print("\nОграничения разрезов:")
        for constraint, expression in self.cut_constraints:
            print(expression)

    def print_cuts(self):
        print("Ограничения разрезов:")
        for cut_set in self.cuts:
            all_islands = [island['id'] for island in self.puzzle.get_all_islands()]
            cut_complement = [island_id for island_id in all_islands if island_id not in cut_set]
            expr_str = ' + '.join(
                [f'x_{min(i, j)},{max(i, j)} + y_{min(i, j)},{max(i, j)}' for i in cut_set for j in cut_complement if
                 (min(i, j), max(i, j)) in self.x])
            print(f"{expr_str} >= 1, {{{', '.join(map(str, cut_set))}}} ⊂ V")
            print()
