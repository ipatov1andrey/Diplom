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
        """Реализует алгоритм локального поиска."""
        if fixed_bridges is None:
            fixed_bridges = set()
        if broken_islands is None:
            broken_islands = set()

        # Инициализируем граф из решения
        self.graph = self.build_graph_from_solution(solution)
        
        while True:
            # Находим компоненты связности
            components = list(nx.connected_components(self.graph))
            
            # Случайная перестановка островов
            islands = list(self.graph.nodes())
            random.shuffle(islands)
            
            for v in islands:
                broken_islands = set()
                fixed_bridges = set()
                
                # Ищем остров в другой компоненте для соединения
                v_component = next(c for c in components if v in c)
                for u in islands:
                    if u not in v_component and self._can_add_bridge(v, u):
                        # Добавляем мост
                        self.graph.add_edge(v, u)
                        fixed_bridges.add((min(v, u), max(v, u)))
                        broken_islands.update([v, u])
                        break
                
                # Исправляем сломанные острова
                while broken_islands:
                    w = broken_islands.pop()
                    w_degree = self._island_degrees[w]
                    current_degree = self.graph.degree(w)
                    
                    if current_degree > w_degree:
                        # Удаляем случайный мост, кроме зафиксированных
                        incident_edges = list(self.graph.edges(w))
                        available_edges = [e for e in incident_edges if (min(e), max(e)) not in fixed_bridges]
                        if available_edges:
                            edge_to_remove = random.choice(available_edges)
                            self.graph.remove_edge(*edge_to_remove)
                            broken_islands.update(edge_to_remove)
                    else:
                        # Пытаемся добавить мост
                        for u in islands:
                            if u != w and self._can_add_bridge(w, u) and (min(w, u), max(w, u)) not in fixed_bridges:
                                self.graph.add_edge(w, u)
                                fixed_bridges.add((min(w, u), max(w, u)))
                                broken_islands.add(u)
                                break
                
                # Проверяем решение
                if self.is_solution_valid():
                    return self.extract_solution()
                
                # Проверяем число компонент
                new_components = list(nx.connected_components(self.graph))
                if len(new_components) < len(components):
                    # Сбрасываем цикл
                    break
                
                # Откатываем изменения
                self.graph = self.build_graph_from_solution(solution)
            
            # Если все острова проверены и улучшений нет
            if len(components) == len(nx.connected_components(self.graph)):
                break
        
        return None

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

    def _repair_broken_islands_fast(self, broken_islands, fixed_bridges, all_islands):
        """Быстрый ремонт сломанных островов."""
        max_repair_attempts = 5
        repair_attempts = 0
        visited_islands = set()
        
        while broken_islands and repair_attempts < max_repair_attempts:
            repair_attempts += 1
            
            # Выбираем остров с наибольшей разницей между текущей и требуемой степенью
            island_to_fix = max(broken_islands, 
                              key=lambda x: abs(self.graph.degree(x) - self._island_degrees[x]))
            
            if island_to_fix in visited_islands:
                continue
                
            visited_islands.add(island_to_fix)
            incident_edges = list(self.graph.edges(island_to_fix, data=True))
            
            island_degree = self._island_degrees[island_to_fix]
            current_degree = sum(data['weight'] for _, _, data in incident_edges)
            
            if current_degree > island_degree:
                # Удаляем мост с наименьшим приоритетом
                removable_edges = [(edge, self._bridge_priorities.get((min(edge[0], edge[1]), max(edge[0], edge[1])), 0))
                                 for edge in incident_edges 
                                 if (min(edge[0], edge[1]), max(edge[0], edge[1])) not in fixed_bridges]
                if not removable_edges:
                    return False
                    
                edge_to_remove = min(removable_edges, key=lambda x: x[1])[0]
                self.graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
                broken_islands.add(edge_to_remove[1])
                
            elif current_degree < island_degree:
                # Пытаемся добавить мост с наивысшим приоритетом
                best_bridge = None
                best_priority = -1
                
                for other in all_islands:
                    if other != island_to_fix and not self.graph.has_edge(island_to_fix, other):
                        bridge = (min(island_to_fix, other), max(island_to_fix, other))
                        if bridge in self._possible_bridges and bridge not in fixed_bridges:
                            if not self._has_intersections_fast(
                                self._island_positions[island_to_fix],
                                self._island_positions[other]
                            ):
                                priority = self._bridge_priorities.get(bridge, 0)
                                if priority > best_priority:
                                    best_priority = priority
                                    best_bridge = (other, bridge)
                
                if best_bridge:
                    other, bridge = best_bridge
                    self.graph.add_edge(island_to_fix, other, weight=1)
                    broken_islands.add(other)
                else:
                    return False
                    
        return len(broken_islands) == 0