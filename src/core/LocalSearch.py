import random
import networkx as nx  #  Для работы с графами
from ortools.linear_solver import pywraplp
from src.core.model import Model
from src.core.solver import Solver

class LocalSearch:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.graph = self.build_graph_from_puzzle(puzzle)
        self.model = Model(puzzle)
        self.solver = Solver(self.model)
        self._bridge_cache = {}
        self._distance_cache = {}
        self._component_cache = {}
        self._island_positions = {island['id']: (island['x'], island['y']) for island in puzzle.get_all_islands()}
        self._grid = puzzle.grid
        self._max_x = len(puzzle.grid[0])
        self._max_y = len(puzzle.grid)
        self._island_degrees = {island['id']: island['degree'] for island in puzzle.get_all_islands()}
        self._neighbors_cache = self._precompute_neighbors()
        self._bridge_positions = set()  # Кэш позиций мостов

    def build_graph_from_puzzle(self, puzzle):
        """Создает граф NetworkX из текущего состояния головоломки."""
        graph = nx.Graph()
        islands = puzzle.get_all_islands()
        for island in islands:
            graph.add_node(island['id'], degree=island['degree'])
        return graph

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

    def local_search(self):
        """Реализует алгоритм локального поиска."""
        max_iterations = 20  # Уменьшаем максимальное количество итераций
        iteration = 0
        visited_states = set()
        no_improvement_count = 0
        max_no_improvement = 3  # Уменьшаем количество итераций без улучшения
        
        # Предварительно вычисляем все возможные мосты
        self._precompute_possible_bridges()
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Local search iteration {iteration}/{max_iterations}")
            
            V = list(self.graph.nodes)
            random.shuffle(V)
            
            initial_components = self._get_components()
            initial_component_count = len(initial_components)
            
            improvement_found = False
            
            # Сортируем узлы по степени для приоритизации
            V.sort(key=lambda x: self._island_degrees[x], reverse=True)
            
            for start_node in V:
                if improvement_found:
                    break
                    
                initial_edges = list(self.graph.edges(data=True))
                broken_islands = set()
                fixed_bridges = set()
                
                components = self._get_components()
                if len(components) > 1:
                    # Используем предварительно вычисленные мосты
                    other_component_node = self._find_nearest_component_fast(start_node, components)
                    
                    if other_component_node is not None:
                        if self._can_add_bridge_fast(start_node, other_component_node):
                            self.graph.add_edge(start_node, other_component_node, weight=1)
                            fixed_bridges.add((start_node, other_component_node))
                            broken_islands.update([start_node, other_component_node])
                            improvement_found = True
                
                if improvement_found:
                    repair_success = self._repair_broken_islands_fast(broken_islands, fixed_bridges, V)
                    if repair_success and nx.is_connected(self.graph) and self.is_valid_solution():
                        return self.extract_solution()
                else:
                    # Откатываем изменения сразу, если улучшения не найдено
                    self.graph.remove_edges_from(list(self.graph.edges()))
                    for u, v, data in initial_edges:
                        self.graph.add_edge(u, v, **data)
            
            current_state = hash(str(self.graph.edges(data=True)))
            if current_state in visited_states:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    break
            else:
                visited_states.add(current_state)
                no_improvement_count = 0
            
            if not improvement_found:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    break
            else:
                no_improvement_count = 0
            
            if len(self._get_components()) == initial_component_count:
                return None
        
        return None

    def _precompute_possible_bridges(self):
        """Предварительно вычисляет все возможные мосты."""
        self._possible_bridges = {}
        islands = self.puzzle.get_all_islands()
        
        for i, island1 in enumerate(islands):
            for island2 in islands[i+1:]:
                if self._can_add_bridge_fast(island1['id'], island2['id']):
                    self._possible_bridges[(island1['id'], island2['id'])] = True

    def _get_components(self):
        """Получает компоненты связности с использованием кэша."""
        components_key = hash(str(self.graph.edges()))
        if components_key in self._component_cache:
            return self._component_cache[components_key]
            
        components = list(nx.connected_components(self.graph))
        self._component_cache[components_key] = components
        return components

    def _find_nearest_component_fast(self, start_node, components):
        """Быстрый поиск ближайшей компоненты с использованием предварительных вычислений."""
        start_pos = self._island_positions[start_node]
        min_distance = float('inf')
        nearest_node = None
        
        # Используем предварительно вычисленных соседей
        for neighbor in self._neighbors_cache[start_node]:
            if neighbor not in components[0]:  # Если сосед в другой компоненте
                if (start_node, neighbor) in self._possible_bridges:
                    distance = self._get_distance_fast(start_node, neighbor)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_node = neighbor
                            
        return nearest_node

    def _get_distance_fast(self, node1, node2):
        """Быстрый расчет расстояния с использованием кэша позиций."""
        cache_key = (min(node1, node2), max(node1, node2))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
            
        pos1 = self._island_positions[node1]
        pos2 = self._island_positions[node2]
        distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        self._distance_cache[cache_key] = distance
        return distance

    def _can_add_bridge_fast(self, island1, island2):
        """Быстрая проверка возможности добавления моста."""
        cache_key = (min(island1, island2), max(island1, island2))
        if cache_key in self._bridge_cache:
            return self._bridge_cache[cache_key]
            
        if cache_key in self._possible_bridges:
            self._bridge_cache[cache_key] = True
            return True
            
        # Проверяем, являются ли острова соседями
        if island2 not in self._neighbors_cache[island1]:
            self._bridge_cache[cache_key] = False
            return False
            
        # Проверяем пересечения
        if self._has_intersections_fast(self._island_positions[island1], self._island_positions[island2]):
            self._bridge_cache[cache_key] = False
            return False
            
        self._bridge_cache[cache_key] = True
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
        if (x1 == x2 and y3 == y4) or (y1 == y2 and x3 == x4):
            if x1 == x2:  # Первый мост вертикальный
                return (min(x3, x4) <= x1 <= max(x3, x4) and
                        min(y1, y2) <= y3 <= max(y1, y2))
            else:  # Первый мост горизонтальный
                return (min(x1, x2) <= x3 <= max(x1, x2) and
                        min(y3, y4) <= y1 <= max(y3, y4))
        return False

    def _repair_broken_islands_fast(self, broken_islands, fixed_bridges, V):
        """Быстрый ремонт сломанных островов."""
        max_repair_attempts = 2  # Уменьшаем количество попыток ремонта
        repair_attempts = 0
        
        while broken_islands and repair_attempts < max_repair_attempts:
            repair_attempts += 1
            island_to_fix = broken_islands.pop()
            incident_edges = list(self.graph.edges(island_to_fix, data=True))
            
            island_degree = self._island_degrees[island_to_fix]
            current_degree = sum(data['weight'] for _, _, data in incident_edges)
            
            if current_degree > island_degree:
                removable_edges = [edge for edge in incident_edges 
                                if (edge[0], edge[1]) not in fixed_bridges and 
                                (edge[1], edge[0]) not in fixed_bridges]
                if not removable_edges:
                    return False
                edge_to_remove = random.choice(removable_edges)
                self.graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
                broken_islands.add(edge_to_remove[1])
                
            elif current_degree < island_degree:
                # Используем предварительно вычисленных соседей
                eligible_neighbors = [node for node in self._neighbors_cache[island_to_fix]
                                   if node != island_to_fix 
                                   and not self.graph.has_edge(island_to_fix, node)
                                   and (island_to_fix, node) in self._possible_bridges]
                                   
                if not eligible_neighbors:
                    return False
                    
                eligible_neighbors.sort(key=lambda n: self._get_distance_fast(island_to_fix, n))
                
                bridge_added = False
                for neighbor in eligible_neighbors:
                    if self._can_add_bridge_fast(island_to_fix, neighbor):
                        self.graph.add_edge(island_to_fix, neighbor, weight=1)
                        broken_islands.add(neighbor)
                        bridge_added = True
                        break
                        
                if not bridge_added:
                    return False
                    
        return len(broken_islands) == 0

    def is_valid_solution(self):
        """Проверяет, является ли текущее состояние графа допустимым решением."""
        # Проверяем связность графа
        if not nx.is_connected(self.graph):
            return False
            
        # Проверяем степень каждого острова
        for node in self.graph.nodes():
            if self.graph.degree(node) != self._island_degrees[node]:
                return False
                
        # Проверяем, что все мосты допустимы
        for edge in self.graph.edges():
            if not self._can_add_bridge_fast(edge[0], edge[1]):
                return False
                
        return True

    def extract_solution(self):
        """Извлекает решение из графа."""
        solution = {}
        for u, v, data in self.graph.edges(data=True):
            solution[(u, v)] = data.get('weight', 1)
        return solution