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
        self.max_iterations = 5
        self._island_positions = {island['id']: (island['x'], island['y']) for island in puzzle.get_all_islands()}
        self._island_degrees = {island['id']: island['degree'] for island in puzzle.get_all_islands()}
        self.fixed_bridges = set()
        self.broken_islands = set()
        self._possible_bridges = self._precompute_possible_bridges()

    def solve(self):
        """Реализует модифицированный итеративный ЦЛП алгоритм."""
        self.model = Model(self.puzzle)
        self.solver = Solver(self.model)
        
        iteration = 0
        has_integer_solution = True
        
        while iteration < self.max_iterations and has_integer_solution:
            iteration += 1
            
            try:
                # Решаем модель ЦЛП
                status, solution = self.solver.solve()
                
                # Проверяем наличие целочисленного решения
                if solution is None:
                    has_integer_solution = False
                    continue

                # Проверяем связность
                if self.is_connected():
                    return solution

                # Добавляем ограничения разрезов для каждой компоненты
                self.add_cut_constraints()

                # Запускаем локальный поиск
                local_search = LocalSearch(self.puzzle)
                local_solution = local_search.local_search(solution, self.fixed_bridges, self.broken_islands)

                if local_solution:
                    if self.is_solution_valid(local_solution):
                        return local_solution

            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                continue

        return None

    def add_cut_constraints(self):
        """Добавляет ограничения разрезов для каждой компоненты связности."""
        try:
            graph = self.build_graph_from_solution()
            # Преобразуем генератор в список
            components = list(nx.connected_components(graph))
            
            if not components:
                return
                
            # Добавляем ограничения для каждой компоненты
            for component in components:
                if len(component) > 0:  # Проверяем, что компонента не пустая
                    self.add_cut_constraint_to_model(component)
                    
        except Exception as e:
            print(f"Error in add_cut_constraints: {str(e)}")
            return

    def add_cut_constraint_to_model(self, component):
        """Добавляет ограничение разреза для компонента."""
        try:
            if not component:
                return False

            # Создаем ограничение для компонента
            constraint = self.model.solver.Add(sum(
                self.model.single_bridge[min(i, j), max(i, j)] + 2 * self.model.double_bridge[min(i, j), max(i, j)]
                for i in component
                for j in range(len(self.puzzle.get_all_islands()))
                if j not in component and (min(i, j), max(i, j)) in self.model.single_bridge
            ) >= 1)
            
            return True
        except Exception as e:
            print(f"Error adding cut constraint: {e}")
            return False

    def is_connected(self):
        """Проверяет, является ли текущее решение связным."""
        graph = self.build_graph_from_solution()
        return nx.is_connected(graph)

    def build_graph_from_solution(self):
        """Создает граф NetworkX на основе текущего решения ЦЛП."""
        solution = self.extract_solution()
        if not solution:
            return nx.Graph()
            
        graph = nx.Graph()
        islands = self.puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'], degree=island['degree'])

        for (i, j), bridges in solution.items():
            graph.add_edge(i, j, weight=bridges)
            
        return graph

    def extract_solution(self):
        """Извлекает решение из солвера."""
        if not self.model:
            return None
            
        solution = {}
        for (i, j) in self._possible_bridges.keys():
            try:
                if (i, j) in self.model.single_bridge and (i, j) in self.model.double_bridge:
                    single = self.model.single_bridge[i, j].solution_value()
                    double = self.model.double_bridge[i, j].solution_value()
                    if single > 0 or double > 0:
                        solution[(i, j)] = single + 2 * double
            except (KeyError, AttributeError):
                continue
        return solution

    def is_solution_valid(self, solution):
        """Проверяет, является ли решение допустимым."""
        # Проверяем степень каждого острова
        for island_id, degree in self._island_degrees.items():
            bridges_count = 0
            for (i, j), bridges in solution.items():
                if i == island_id or j == island_id:
                    bridges_count += bridges
                    
            if bridges_count != degree:
                return False
                
        # Проверяем, что все мосты допустимы
        for (i, j) in solution.keys():
            if not self._can_add_bridge_fast(i, j):
                return False
                
        return True

    def _can_add_bridge_fast(self, island1, island2):
        """Быстрая проверка возможности добавления моста."""
        pos1 = self._island_positions[island1]
        pos2 = self._island_positions[island2]
        
        # Проверяем, что острова на одной линии
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            return False
            
        # Проверяем, нет ли других островов между ними
        if pos1[0] == pos2[0]:  # Вертикальный мост
            start_y = min(pos1[1], pos2[1]) + 1
            end_y = max(pos1[1], pos2[1])
            for y in range(start_y, end_y):
                if self.puzzle.grid[y][pos1[0]] != 0:
                    return False
        else:  # Горизонтальный мост
            start_x = min(pos1[0], pos2[0]) + 1
            end_x = max(pos1[0], pos2[0])
            for x in range(start_x, end_x):
                if self.puzzle.grid[pos1[1]][x] != 0:
                    return False
                    
        return True

    def _precompute_possible_bridges(self):
        """Предварительно вычисляет все возможные мосты."""
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
        # Получаем все острова
        islands = self.model.puzzle.get_all_islands()
        # S - множество узлов в компоненте
        S = set(component)
        # V\S - множество узлов вне компоненты
        V_minus_S = [v['id'] for v in islands if v['id'] not in S]

        # Создаем ограничение: сумма x_ij >= 1 для i из S и j из V\S
        # где x_ij - количество мостов между островами i и j
        constraint_expression = sum(
            self.model.x[min(i, j), max(i, j)] 
            for i in S 
            for j in V_minus_S 
            if (min(i, j), max(i, j)) in self.model.x
        )
        
        # Добавляем ограничение в модель
        constraint = self.model.solver.Add(constraint_expression >= 1)
        # Сохраняем ограничение
        self.cut_constraints.append(constraint)

    def remove_cut_constraints(self):
        """Удаляет все добавленные ограничения разрезов."""
        for constraint in self.cut_constraints:
            self.model.solver.Remove(constraint)
        self.cut_constraints.clear()
