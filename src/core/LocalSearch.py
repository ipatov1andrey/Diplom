import random
import networkx as nx  #  Для работы с графами
from ortools.linear_solver import pywraplp
from src.core.model import Model
from src.core.solver import Solver
from src.core.puzzle import Puzzle

class LocalSearch:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.graph = nx.Graph()
        self._initialize_graph()
        self._island_positions = {island['id']: (island['x'], island['y']) for island in puzzle.get_all_islands()}
        self._grid = puzzle.grid
        self._max_x = len(puzzle.grid[0])
        self._max_y = len(puzzle.grid)
        self._island_degrees = {island['id']: island['degree'] for island in puzzle.get_all_islands()}
        self._neighbors_cache = self._precompute_neighbors()
        self._possible_bridges = self._precompute_possible_bridges()
        self._bridge_priorities = self._compute_bridge_priorities()
        self.current_solution = None  # Добавляем инициализацию current_solution

    def _initialize_graph(self):
        """Инициализирует граф с островами."""
        islands = self.puzzle.get_all_islands()
        for island in islands:
            self.graph.add_node(island['id'], degree=island['degree'])

    def _precompute_neighbors(self):
        """Предварительно вычисляет соседей для каждого острова."""
        neighbors = {}
        islands = self.puzzle.get_all_islands()
        
        for island in islands:
            island_id = island['id']
            neighbors[island_id] = []
            x, y = island['x'], island['y']
            
            # Проверяем соседей по горизонтали
            for dx in [-1, 1]:
                new_x = x + dx
                while 0 <= new_x < self._max_x:
                    if self._grid[y][new_x] != 0:
                        neighbors[island_id].append(self._grid[y][new_x])
                        break
                    new_x += dx
                    
            # Проверяем соседей по вертикали
            for dy in [-1, 1]:
                new_y = y + dy
                while 0 <= new_y < self._max_y:
                    if self._grid[new_y][x] != 0:
                        neighbors[island_id].append(self._grid[new_y][x])
                        break
                    new_y += dy
                    
        return neighbors

    def _precompute_possible_bridges(self):
        """Предварительно вычисляет все возможные мосты."""
        possible_bridges = {}
        islands = self.puzzle.get_all_islands()
        
        for i, island1 in enumerate(islands):
            for island2 in islands[i+1:]:
                if self._can_add_bridge(island1['id'], island2['id']):
                    possible_bridges[(island1['id'], island2['id'])] = True

        return possible_bridges

    def _can_add_bridge(self, island1, island2):
        """Проверяет возможность добавления моста."""
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
                if self._grid[y][pos1[0]] != 0:
                    return False
        else:  # Горизонтальный мост
            start_x = min(pos1[0], pos2[0]) + 1
            end_x = max(pos1[0], pos2[0])
            for x in range(start_x, end_x):
                if self._grid[pos1[1]][x] != 0:
                    return False
                    
        return True

    def _compute_bridge_priorities(self):
        """Вычисляет приоритеты для мостов на основе нескольких факторов."""
        priorities = {}
        islands = self.puzzle.get_all_islands()
        
        for i, island1 in enumerate(islands):
            for island2 in islands[i+1:]:
                if (island1['id'], island2['id']) in self._possible_bridges:
                    # Вычисляем различные факторы
                    distance = abs(island1['x'] - island2['x']) + abs(island1['y'] - island2['y'])
                    degree_sum = island1['degree'] + island2['degree']
                    degree_diff = abs(island1['degree'] - island2['degree'])
                    
                    # Приоритет зависит от:
                    # 1. Суммы степеней (чем больше, тем лучше)
                    # 2. Расстояния (чем меньше, тем лучше)
                    # 3. Разницы степеней (чем меньше, тем лучше)
                    priority = (degree_sum * 2) / (distance + 1) - degree_diff
                    
                    # Учитываем количество возможных мостов
                    possible_bridges = len([b for b in self._possible_bridges 
                                         if b[0] in [island1['id'], island2['id']] or 
                                            b[1] in [island1['id'], island2['id']]])
                    priority *= (1 + 1/possible_bridges)
                    
                    priorities[(island1['id'], island2['id'])] = priority
                    
        return priorities

    def local_search(self, solution, fixed_bridges=None, broken_islands=None):
        """Выполняет локальный поиск для улучшения решения согласно Алгоритму 2."""
        if not solution:
            return None

        self.current_solution = solution.copy()
        self.graph = self.build_graph_from_solution(solution)
        if not self.graph or not self.graph.nodes():
            return None

        # Получаем список всех островов
        all_islands = list(self.graph.nodes())

        while True:  # Основной цикл repeat-until
            # Находим компоненты связности
            components = list(nx.connected_components(self.graph)) # Ensure components is a list
            num_components_before = len(components)
            if num_components_before == 1:
                return self.current_solution

            # Перемешиваем острова для случайного порядка
            random.shuffle(all_islands)

            found_improvement_in_for_loop = False # Флаг для сброса цикла for

            for v in all_islands:  # Для каждого острова v в порядке перестановки
                # Инициализируем множества для фиксированных мостов и сломанных островов для каждой попытки с v
                fixed_bridges = set()
                broken_islands = set()
                added_bridge = None # Переменная для хранения добавленного моста (u, v)

                # Находим компоненту, содержащую v
                v_component = next(comp for comp in components if v in comp)

                # Ищем острова в других компонентах
                other_components = [comp for comp in components if comp != v_component]

                # Пытаемся найти подходящий остров u в другой компоненте
                bridge_found_between_components = False
                for other_comp in other_components:
                    for u in other_comp:
                        if self._can_add_bridge_fast(u, v):
                            # Добавляем мост между u и v
                            self.graph.add_edge(u, v, weight=1)
                            bridge_key = (min(u, v), max(u, v))
                            self.current_solution[bridge_key] = self.current_solution.get(bridge_key, 0) + 1 # Increment bridge count
                            added_bridge = (u, v)

                            # Обновляем множества
                            fixed_bridges.add(bridge_key)
                            broken_islands.update([u, v])

                            bridge_found_between_components = True
                            break # Выходим из внутренних циклов после добавления моста
                    if bridge_found_between_components:
                            break

                # Если мост между компонентами добавлен, пытаемся починить
                if added_bridge:
                    # Передаем лимит попыток ремонта
                    repair_successful = self._repair_broken_islands(broken_islands, fixed_bridges, list(self.graph.nodes()), max_repair_attempts=150)

                    # Проверка после цикла while brokenIslands
                    if self.is_solution_valid():
                        return self.current_solution # Найдено решение головоломки

                    # Проверяем, уменьшилось ли число компонент
                    new_components = list(nx.connected_components(self.graph))
                    num_components_after = len(new_components)

                    if num_components_after < num_components_before:
                        # Число компонент уменьшилось, сбрасываем цикл for и возвращаемся в repeat
                        found_improvement_in_for_loop = True
                        break # Выход из цикла for v
                    elif not repair_successful: # Если ремонт не удался и нет улучшения компонент
                        # Откатываем изменения
                        self._rollback_changes(added_bridge[0], added_bridge[1])
                        pass # Просто продолжаем цикл for
                    else:
                        # Число компонент не уменьшилось, но ремонт был успешным (попытки не кончились)
                        # Откатываем изменения
                        self._rollback_changes(added_bridge[0], added_bridge[1])
                        pass # Просто продолжаем цикл for

            # Если не нашли улучшений (уменьшения числа компонент) после проверки всех островов, выходим из repeat
            if not found_improvement_in_for_loop:
                        break

            return self.current_solution

    def _repair_broken_islands(self, broken_islands, fixed_bridges, all_islands, max_repair_attempts=150):
        """Ремонтирует сломанные острова согласно алгоритму."""
        repair_attempts = 0
        while broken_islands and repair_attempts < max_repair_attempts:
            repair_attempts += 1
            # Выбираем сломанный остров w
            w = broken_islands.pop()
            
            # Получаем текущую степень острова
            current_degree = sum(data.get('weight', 1) for _, _, data in self.graph.edges(w, data=True))
            target_degree = self._island_degrees.get(w, 0)
            
            if current_degree > target_degree:
                # Если степень больше требуемой, удаляем случайный мост
                removable_edges = [
                    (edge, self._bridge_priorities.get((min(edge[0], edge[1]), max(edge[0], edge[1])), 0))
                    for edge in list(self.graph.edges(w, data=True))
                    if (min(edge[0], edge[1]), max(edge[0], edge[1])) not in fixed_bridges
                ]
                
                if not removable_edges:
                    return False

                # Выбираем мост с наименьшим приоритетом
                edge_to_remove = min(removable_edges, key=lambda x: x[1])[0]
                if self.graph.has_edge(edge_to_remove[0], edge_to_remove[1]):
                    self.graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
                bridge_key = (min(edge_to_remove[0], edge_to_remove[1]),
                              max(edge_to_remove[0], edge_to_remove[1]))
                if bridge_key in self.current_solution:
                    del self.current_solution[bridge_key]
                
                # Добавляем соседний остров в список сломанных
                other_island = edge_to_remove[1] if edge_to_remove[0] == w else edge_to_remove[0]
                broken_islands.add(other_island)
                
            elif current_degree < target_degree:
                # Если степень меньше требуемой, пытаемся добавить мост
                best_bridge = None
                best_priority = -1
                
                for other in list(all_islands):
                    if other != w and not self.graph.has_edge(w, other):
                        bridge = (min(w, other), max(w, other))
                        if bridge in self._possible_bridges and bridge not in fixed_bridges:
                            if not self._has_intersections_fast(
                                self._island_positions[w],
                                self._island_positions[other]
                            ):
                                priority = self._bridge_priorities.get(bridge, 0)
                                if priority > best_priority:
                                    best_priority = priority
                                    best_bridge = (other, bridge)
                
                if best_bridge:
                    other, bridge = best_bridge
                    self.graph.add_edge(w, other, weight=1)
                    self.current_solution[bridge] = 1
                    broken_islands.add(other)
                else:
                    return False

            return True

    def _rollback_changes(self, u, v):
        """Откатывает изменения, связанные с добавлением моста между u и v."""
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
        bridge_key = (min(u, v), max(u, v))
        if bridge_key in self.current_solution:
            del self.current_solution[bridge_key]

    def build_graph_from_solution(self, solution):
        """Создает граф NetworkX на основе решения."""
        graph = nx.Graph()
        islands = self.puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'], degree=island['degree'])

        for (i, j), bridges in solution.items():
            graph.add_edge(i, j, weight=bridges)
            
        return graph

    def extract_solution(self):
        """Извлекает решение из графа."""
        solution = {}
        for i, j in self.graph.edges():
            solution[(min(i, j), max(i, j))] = self.graph[i][j].get('weight', 1)
        return solution

    def is_solution_valid(self):
        """Проверяет, является ли решение допустимым."""
        # Проверяем степень каждого острова
        for island_id, degree in self._island_degrees.items():
            if self.graph.degree(island_id) != degree:
                return False
                
        # Проверяем, что все мосты допустимы
        for i, j in self.graph.edges():
            if not self._can_add_bridge(i, j):
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
                if self._grid[y][pos1[0]] != 0:
                    return False
        else:  # Горизонтальный мост
            start_x = min(pos1[0], pos2[0]) + 1
            end_x = max(pos1[0], pos2[0])
            for x in range(start_x, end_x):
                if self._grid[pos1[1]][x] != 0:
                    return False
                    
        return True

    def _has_intersections_fast(self, pos1, pos2):
        """Быстрая проверка пересечений."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Проверяем только существующие мосты
        for edge in self.graph.edges():
            if edge == (pos1, pos2) or edge == (pos2, pos1):
                continue
                
            edge_pos1 = self._island_positions[edge[0]]
            edge_pos2 = self._island_positions[edge[1]]
            
            if self._bridges_intersect_fast(x1, y1, x2, y2, edge_pos1[0], edge_pos1[1], edge_pos2[0], edge_pos2[1]):
                return True
                
        return False

    def _bridges_intersect_fast(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Быстрая проверка пересечения мостов."""
        # Проверяем, что мосты перпендикулярны
        if (x1 == x2 and y3 == y4) or (y1 == y2 and x3 == x4):
            if x1 == x2:  # Первый мост вертикальный
                return (min(x3, x4) <= x1 <= max(x3, x4) and
                        min(y1, y2) <= y3 <= max(y1, y2))
            else:  # Первый мост горизонтальный
                return (min(x1, x2) <= x3 <= max(x1, x2) and
                        min(y3, y4) <= y1 <= max(y3, y4))
            return False