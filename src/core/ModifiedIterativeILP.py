from ortools.linear_solver import pywraplp #  Уже должно быть импортировано
from src.core.model import Model
from src.core.solver import Solver
from src.core.LocalSearch import LocalSearch  # Добавляем импорт LocalSearch
import networkx as nx  # Добавляем импорт для работы с графами
#  Импортируем LocalSearch
# from local_search import LocalSearch

class ModifiedIterativeILP:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.model = None
        self.solver = None
        self.cut_constraints = []
        self.max_iterations = 40
        self._island_positions = {island['id']: (island['x'], island['y']) for island in puzzle.get_all_islands()}
        self._island_degrees = {island['id']: island['degree'] for island in puzzle.get_all_islands()}
        self.fixed_bridges = set()
        self.broken_islands = set()
        self._possible_bridges = self._precompute_possible_bridges()

    def solve(self):
        """Реализует модифицированный итеративный ЦЛП алгоритм. Возвращает (solution, iteration_count, algo_time)."""
        import time
        self.model = Model(self.puzzle)
        self.solver = Solver(self.model)
        iteration = 0
        algo_time = 0.0
        while iteration < self.max_iterations:
            iteration += 1
            try:
                # Решаем ILP
                solver_start = time.time()
                status, solution = self.solver.solve()
                solver_end = time.time()
                algo_time += solver_end - solver_start

                if solution is None:
                    # Нет целочисленного решения на этой итерации
                    return None, iteration, algo_time

                # Проверяем связность/валидность ILP-решения
                graph_from_ilp = self.build_graph_from_solution()
                if nx.is_connected(graph_from_ilp) and self.is_solution_valid(solution):
                    return solution, iteration, algo_time

                # Попытка локального поиска перед добавлением разрезов
                local_search = LocalSearch(self.puzzle)
                local_solution = local_search.local_search(solution, self.fixed_bridges, self.broken_islands)
                if local_solution:
                    # Быстрая проверка локального решения
                    if self.is_solution_valid(local_solution):
                        # Строим граф из локального решения для проверки связности
                        G = nx.Graph()
                        for island in self.puzzle.get_all_islands():
                            G.add_node(island['id'])
                        for (i, j), bridges in local_solution.items():
                            if bridges > 0:
                                G.add_edge(i, j)
                        if nx.is_connected(G):
                            return local_solution, iteration, algo_time

                # Если локальный поиск не помог — добавляем разрезы и повторяем
                self.add_cut_constraints()

            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                continue
        return None, iteration, algo_time

    def add_cut_constraints(self):
        """Добавляет ограничения разрезов для каждой компоненты связности."""
        try:
            graph = self.build_graph_from_solution()
            components = list(nx.connected_components(graph))
            if not components:
                return
            for component in components:
                if len(component) > 0:
                    self.add_cut_constraint_to_model(list(component))
        except Exception as e:
            print(f"Error in add_cut_constraints: {str(e)}")
            return

    def add_cut_constraint_to_model(self, component):
        """Добавляет ограничение разреза для компонента."""
        try:
            if not component:
                return False
            constraint = self.model.solver.Add(sum(
                self.model.single_bridge[min(i, j), max(i, j)]
                for i in component
                for j in range(len(self.puzzle.get_all_islands()))
                if j not in component and (min(i, j), max(i, j)) in self.model.single_bridge
            ) >= 1)
            return True
        except Exception as e:
            print(f"Error adding cut constraint: {e}")
            return False

    def is_connected(self):
        graph = self.build_graph_from_solution()
        return nx.is_connected(graph)

    def build_graph_from_solution(self):
        solution = self.extract_solution()
        if not solution:
            return nx.Graph()
        graph = nx.Graph()
        islands = self.puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'], degree=island['degree'])
        for (i, j), bridges in solution.items():
            if graph.has_node(i) and graph.has_node(j):
                graph.add_edge(i, j, weight=bridges)
        return graph

    def extract_solution(self):
        if not self.model:
            return None
        solution = {}
        for (i, j) in list(self._possible_bridges.keys()):
            try:
                if (i, j) in self.model.single_bridge and (i, j) in self.model.double_bridge:
                    single_val = self.model.single_bridge[i, j].SolutionValue()
                    double_val = self.model.double_bridge[i, j].SolutionValue()
                    if single_val > 1e-9 or double_val > 1e-9:
                        bridge_count = int(round(single_val)) + int(round(double_val))
                        if bridge_count > 0:
                            solution[(i, j)] = bridge_count
            except (KeyError, AttributeError, TypeError) as e:
                print(f"Error extracting solution value for bridge ({i}, {j}): {e}")
                continue
        return solution

    def is_solution_valid(self, solution):
        for island_id, degree in self._island_degrees.items():
            bridges_count = 0
            for (i, j), bridges in solution.items():
                if i == island_id or j == island_id:
                    bridges_count += bridges
            if bridges_count != degree:
                return False
        for (i, j) in solution.keys():
            if not self._can_add_bridge_fast(i, j):
                return False
        return True

    def _can_add_bridge_fast(self, island1, island2):
        pos1 = self._island_positions[island1]
        pos2 = self._island_positions[island2]
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            return False
        if pos1[0] == pos2[0]:
            start_y = min(pos1[1], pos2[1]) + 1
            end_y = max(pos1[1], pos2[1])
            for y in range(start_y, end_y):
                if self.puzzle.grid[y][pos1[0]] != 0:
                    return False
        else:
            start_x = min(pos1[0], pos2[0]) + 1
            end_x = max(pos1[0], pos2[0])
            for x in range(start_x, end_x):
                if self.puzzle.grid[pos1[1]][x] != 0:
                    return False
        return True

    def _precompute_possible_bridges(self):
        possible_bridges = {}
        islands = self.puzzle.get_all_islands()
        for i, island1 in enumerate(islands):
            for island2 in islands[i+1:]:
                if self._can_add_bridge_fast(island1['id'], island2['id']):
                    possible_bridges[(island1['id'], island2['id'])] = True
        return possible_bridges

class ModifiedIterativeOriginalILP(ModifiedIterativeILP):
    """Реализация модифицированного итеративного ЦЛП алгоритма для оригинальной модели."""
    def __init__(self, puzzle):
        super().__init__(puzzle)
        self.cut_constraints = []

    def add_cut_constraint_to_model(self, component):
        """Добавляет ограничение разреза в оригинальную модель."""
        islands = self.model.puzzle.get_all_islands()
        S = set(component)
        V_minus_S = [v['id'] for v in islands if v['id'] not in S]
        constraint_expression = sum(
            self.model.single_bridge[min(i, j), max(i, j)] 
            for i in S 
            for j in V_minus_S 
            if (min(i, j), max(i, j)) in self.model.single_bridge
        )
        constraint = self.model.solver.Add(constraint_expression >= 1)
        self.cut_constraints.append(constraint)

    def remove_cut_constraints(self):
        for constraint in self.cut_constraints:
            self.model.solver.Remove(constraint)
        self.cut_constraints.clear()
