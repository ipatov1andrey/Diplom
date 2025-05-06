from ortools.linear_solver import pywraplp #  Уже должно быть импортировано
from src.core.model import Model
from src.core.solver import Solver
from src.core.LocalSearch import LocalSearch  # Добавляем импорт LocalSearch
import networkx as nx  # Добавляем импорт для работы с графами
#  Импортируем LocalSearch
# from local_search import LocalSearch

class ModifiedIterativeILP: # Алгоритм 3
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.model = None # Ссылка на модель ЦЛП
        self.solver = None
        self.cut_constraints = []
        self.max_iterations = 10  # Assuming a default max_iterations
        self._solution_cache = {}  # Кэш для решений
        self._graph_cache = {}  # Кэш для графов
        self._component_cache = {}  # Кэш для компонент
        self._island_positions = {island['id']: (island['x'], island['y']) for island in puzzle.get_all_islands()}
        self._island_degrees = {island['id']: island['degree'] for island in puzzle.get_all_islands()}
        self._precompute_possible_bridges()

    def _precompute_possible_bridges(self):
        """Предварительно вычисляет все возможные мосты."""
        self._possible_bridges = {}
        islands = self.puzzle.get_all_islands()
        
        for i, island1 in enumerate(islands):
            for island2 in islands[i+1:]:
                if self._can_add_bridge_fast(island1['id'], island2['id']):
                    self._possible_bridges[(island1['id'], island2['id'])] = True

    def _can_add_bridge_fast(self, island1, island2):
        """Быстрая проверка возможности добавления моста."""
        cache_key = (min(island1, island2), max(island1, island2))
        if cache_key in self._possible_bridges:
            return self._possible_bridges[cache_key]
            
        pos1 = self._island_positions[island1]
        pos2 = self._island_positions[island2]
        
        # Проверяем, что острова на одной линии
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            self._possible_bridges[cache_key] = False
            return False
            
        # Проверяем, нет ли других островов между ними
        if pos1[0] == pos2[0]:  # Вертикальный мост
            start_y = min(pos1[1], pos2[1]) + 1
            end_y = max(pos1[1], pos2[1])
            for y in range(start_y, end_y):
                if self.puzzle.grid[y][pos1[0]] != 0:
                    self._possible_bridges[cache_key] = False
                    return False
        else:  # Горизонтальный мост
            start_x = min(pos1[0], pos2[0]) + 1
            end_x = max(pos1[0], pos2[0])
            for x in range(start_x, end_x):
                if self.puzzle.grid[pos1[1]][x] != 0:
                    self._possible_bridges[cache_key] = False
                    return False
                    
        self._possible_bridges[cache_key] = True
        return True

    def solve(self):
        """Реализует модифицированный итеративный ЦЛП алгоритм."""
        self.model = Model(self.puzzle)
        self.solver = Solver(self.model)
        
        iteration = 0
        previous_solutions = set()
        local_search_attempts = 0
        max_local_search_attempts = 3
        no_improvement_count = 0
        max_no_improvement = 3
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}/{self.max_iterations}")
            
            try:
                # Решаем модель ЦЛП
                print("Solving ILP model...")
                status, solution = self.solver.solve()
                if solution is None:
                    print("No solution found by solver")
                    return None

                # Проверяем кэш решений
                solution_key = hash(str(solution))
                if solution_key in self._solution_cache:
                    print("Using cached solution")
                    solution = self._solution_cache[solution_key]
                    no_improvement_count += 1
                    if no_improvement_count >= max_no_improvement:
                        print("No improvement for several iterations, trying local search...")
                        local_search_attempts += 1
                        if local_search_attempts >= max_local_search_attempts:
                            print("Too many local search attempts, stopping...")
                            return None
                        no_improvement_count = 0
                else:
                    previous_solutions.add(solution_key)
                    self._solution_cache[solution_key] = solution
                    no_improvement_count = 0
                    local_search_attempts = 0

                # Проверяем связность
                if self.is_connected():
                    print("Connected solution found!")
                    return solution

                # Добавляем ограничения разрезов
                print("Adding cut constraints...")
                self.add_cut_constraints()

                # Локальный поиск
                print("Starting local search...")
                local_search = LocalSearch(self.puzzle)
                local_solution = local_search.local_search()

                if local_solution:
                    print("Local search found a solution!")
                    if self.is_solution_valid(local_solution):
                        print("Local search solution is valid!")
                        return local_solution
                    else:
                        print("Local search solution is invalid, continuing...")
                        self.add_cut_constraints()

            except Exception as e:
                print(f"Error during iteration {iteration}: {str(e)}")
                continue

        print("Maximum iterations reached")
        return None

    def extract_solution(self):
        """Извлекает решение из солвера."""
        if not self.model:
            return None
            
        solution = {}
        for (i, j) in self._possible_bridges.keys():
            try:
                # Проверяем существование переменных в модели
                if (i, j) in self.model.single_bridge and (i, j) in self.model.double_bridge:
                    single = self.model.single_bridge[i, j].solution_value()
                    double = self.model.double_bridge[i, j].solution_value()
                    if single > 0 or double > 0:
                        solution[(i, j)] = single + 2 * double
            except (KeyError, AttributeError):
                continue
        return solution

    def is_connected(self):
        """Проверяет, является ли текущее решение связным."""
        solution = self.extract_solution()
        if not solution:
            return False
            
        # Проверяем кэш графов
        solution_key = hash(str(solution))
        if solution_key in self._graph_cache:
            return nx.is_connected(self._graph_cache[solution_key])
            
        graph = self.build_graph_from_solution()
        self._graph_cache[solution_key] = graph
        return nx.is_connected(graph)

    def build_graph_from_solution(self):
        """Создает граф NetworkX на основе текущего решения ЦЛП."""
        solution = self.extract_solution()
        if not solution:
            return nx.Graph()
            
        # Проверяем кэш графов
        solution_key = hash(str(solution))
        if solution_key in self._graph_cache:
            return self._graph_cache[solution_key]
            
        graph = nx.Graph()
        islands = self.puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'], degree=island['degree'])

        for (i, j), bridges in solution.items():
            graph.add_edge(i, j, weight=bridges)
            
        self._graph_cache[solution_key] = graph
        return graph

    def apply_solution_to_graph(self, solution):
        """Применяет решение (из ЦЛП или локального поиска) к графу LocalSearch."""
        if not hasattr(self, 'local_search'):
            self.local_search = LocalSearch(self.puzzle)

        graph = self.local_search.graph
        graph.remove_edges_from(list(graph.edges()))

        for (i, j), bridges in solution.items():
            graph.add_edge(i, j, weight=bridges)

    def add_cut_constraints(self):
        """Добавляет ограничения разрезов для каждой компоненты связности."""
        try:
            graph = self.build_graph_from_solution()
            components = self._get_components(graph)
            
            if not components:
                print("No components found to add cut constraints")
                return
                
            print(f"Found {len(components)} components to process")
            
            for i, component in enumerate(components):
                try:
                    self.add_cut_constraint_to_model(component)
                    print(f"Added cut constraint for component {i+1}/{len(components)}")
                except Exception as e:
                    print(f"Error adding cut constraint for component {i+1}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in add_cut_constraints: {str(e)}")
            return

    def _get_components(self, graph):
        """Получает компоненты связности с использованием кэша."""
        components_key = hash(str(graph.edges()))
        if components_key in self._component_cache:
            return self._component_cache[components_key]
            
        components = list(nx.connected_components(graph))
        self._component_cache[components_key] = components
        return components

    def add_cut_constraint_to_model(self, component):
        """Добавляет ограничение разреза в модель ЦЛП."""
        try:
            islands = self.puzzle.get_all_islands()
            S = set(component)
            V_minus_S = [v['id'] for v in islands if v['id'] not in S]
            
            if not V_minus_S:
                print("No vertices outside component, skipping constraint")
                return
                
            # Создаем ограничение: сумма (single_bridge + double_bridge) >= 1
            constraint_terms = []
            for i in S:
                for j in V_minus_S:
                    bridge_key = (min(i, j), max(i, j))
                    if bridge_key in self._possible_bridges:
                        try:
                            if bridge_key in self.model.single_bridge and bridge_key in self.model.double_bridge:
                                term = (self.model.single_bridge[bridge_key] + 
                                       self.model.double_bridge[bridge_key])
                                constraint_terms.append(term)
                        except (KeyError, AttributeError) as e:
                            print(f"Warning: Could not add bridge {bridge_key} to constraint: {str(e)}")
                            continue
            
            if not constraint_terms:
                print("No valid bridge terms found for constraint")
                return
                
            constraint_expression = sum(constraint_terms)
            constraint = self.model.solver.Add(constraint_expression >= 1)
            self.cut_constraints.append(constraint)
            print(f"Added cut constraint with {len(constraint_terms)} terms")
            
        except Exception as e:
            print(f"Error in add_cut_constraint_to_model: {str(e)}")
            raise

    def remove_cut_constraints(self):
        """Удаляет все добавленные ограничения разрезов."""
        for constraint in self.cut_constraints:
            self.model.solver.Remove(constraint)
        self.cut_constraints.clear()

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
