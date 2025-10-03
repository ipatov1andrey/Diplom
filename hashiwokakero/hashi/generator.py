
from ortools.linear_solver import pywraplp
from .node import Node, direction_to_vector, is_in_grid
from random import randint, choice, choices, shuffle
from .export import save_grid
from src.utils import is_solution_valid
import os
from datetime import datetime
import sys
import time
import re
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hashiwokakero')))

from src.core.puzzle import Puzzle
from src.core.model_flow import FlowModel
from src.core.solver_flow import FlowSolver
from src.core.model import Model as OriginalModel
from src.core.solver import Solver as OriginalSolver
from src.core.ModifiedIterativeILP import ModifiedIterativeILP
from collections import defaultdict

def generate_solvable_puzzle(w: int, h: int, target_islands: int, target_double_bridge_percentage: int = 0, target_degree_7: int = 0, target_degree_8: int = 0, connectivity_factor_alpha: int = 0) -> list[list[Node]]:
    """
    Генерирует корректную головоломку Hashiwokakero с гарантированным решением.
    """
    start_time = time.time()
    print(f"DEBUG: Генерация пазла {w}x{h}, {target_islands} островов, {target_double_bridge_percentage}% двойных мостов")
    
    # Создаем пустую сетку
    grid = [[Node(i, j) for j in range(h)] for i in range(w)]
    islands = []
    island_coords = []
    deg7_centers = []

    max_bridge_length = min(5, max(2, min(w, h) // 2))
    
    grid = [[Node(i, j) for j in range(h)] for i in range(w)]
    islands = []
    island_coords = []
    possible_island_positions = [(i, j) for i in range(0, w) for j in range(0, h) if grid[i][j].n_type == 0]
    shuffle(possible_island_positions)
    
    def count_potential_neighbors(x, y):
        count = 0
        for dx, dy in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            nx, ny = x + dx, y + dy
            while is_in_grid(nx, ny, w, h) and grid[nx][ny].n_type == 0:
                nx += dx
                ny += dy
            if is_in_grid(nx, ny, w, h):
                count += 1
        return count
    
    if target_double_bridge_percentage > 60:
        possible_island_positions.sort(key=lambda pos: count_potential_neighbors(pos[0], pos[1]))
    elif target_double_bridge_percentage < 40:
        possible_island_positions.sort(key=lambda pos: count_potential_neighbors(pos[0], pos[1]), reverse=True)
    else:
        shuffle(possible_island_positions)
        
    placed_islands_count = 0
    for x, y in possible_island_positions:
        if placed_islands_count >= target_islands:
            break
        grid[x][y].make_island(0)
        islands.append(grid[x][y])
        island_coords.append((x, y))
        placed_islands_count += 1

    print(f"DEBUG: Размещено {placed_islands_count} островов из {target_islands}")

    if placed_islands_count < target_islands:
        empty_cells = [(i, j) for i in range(w) for j in range(h) if grid[i][j].n_type == 0]
        shuffle(empty_cells)
        for i, j in empty_cells:
            if len(islands) >= target_islands:
                break
            grid[i][j].make_island(0)
            islands.append(grid[i][j])
            island_coords.append((i, j))
    island_map = {(x,y): grid[x][y] for x, y in island_coords}
    parent = {coord: coord for coord in island_coords}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        parent[find(u)] = find(v)
    
    edges = []
    for i, (x1, y1) in enumerate(island_coords):
        for j, (x2, y2) in enumerate(island_coords):
            if i >= j: continue
            if x1 == x2:  # Вертикальные
                dist = abs(y1 - y2)
                if dist <= max_bridge_length and all(grid[x1][y].n_type == 0 for y in range(min(y1, y2)+1, max(y1, y2))):
                    edges.append(((x1, y1), (x2, y2), dist, 1))
            elif y1 == y2:  # Горизонтальные
                dist = abs(x1 - x2)
                if dist <= max_bridge_length and all(grid[x][y1].n_type == 0 for x in range(min(x1, x2)+1, max(x1, x2))):
                    edges.append(((x1, y1), (x2, y2), dist, 0))
    
    edges.sort(key=lambda e: e[2])
    print(f"DEBUG: Найдено {len(edges)} возможных рёбер для MST")
    
    bridge_counts = {}
    # Шаг 1: выбираем рёбра MST без рисования, контролируя пересечения путей
    mst_edges = []
    occupied_bridge_cells = set()
    initial_single_bridges = 0
    
    def iter_path_cells(p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        if x1 == x2:
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                yield (x1, y)
        else:
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                yield (x, y1)
    
    for u_coord, v_coord, dist, direction in edges:
        if find(u_coord) == find(v_coord):
            continue
        # Проверяем, что путь для этого ребра не пересекает ранее выбранные пути
        path_conflict = False
        for cell in iter_path_cells(u_coord, v_coord):
            if cell in occupied_bridge_cells:
                path_conflict = True
                break
        if path_conflict:
            continue
        # Добавляем рёбра и помечаем клетки пути как занятые
        mst_edges.append((u_coord, v_coord, direction))
        for cell in iter_path_cells(u_coord, v_coord):
            occupied_bridge_cells.add(cell)
        union(u_coord, v_coord)
        initial_single_bridges += 1

    # Шаг 2: рисуем выбранные рёбра MST
    for u_coord, v_coord, direction in mst_edges:
        x1, y1 = u_coord
        x2, y2 = v_coord
        if x1 == x2:
                for y in range(min(y1, y2)+1, max(y1, y2)):
                    grid[x1][y].make_bridge(1, 1)
        else:
                for x in range(min(x1, x2)+1, max(x1, x2)):
                    grid[x][y1].make_bridge(1, 0)
        bridge_counts[(tuple(sorted((u_coord, v_coord))), direction)] = 1
        island_map[u_coord].i_count += 1
        island_map[v_coord].i_count += 1
 
    print(f"DEBUG: Построен MST с {initial_single_bridges} мостами")

    def path_empty_same_row_or_col(p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        if x1 == x2:
            for y in range(min(y1, y2)+1, max(y1, y2)):
                if grid[x1][y].n_type != 0:
                    return False
            return True
        if y1 == y2:
            for x in range(min(x1, x2)+1, max(x1, x2)):
                if grid[x][y1].n_type != 0:
                    return False
            return True
        return False

    all_edges_unlimited = []
    for i, u in enumerate(island_coords):
        for v in island_coords[i+1:]:
            if find(u) != find(v) and path_empty_same_row_or_col(u, v):
                dist = abs(u[0]-v[0]) + abs(u[1]-v[1])
                direction = 1 if u[0] == v[0] else 0
                all_edges_unlimited.append((u, v, dist, direction))
    all_edges_unlimited.sort(key=lambda e: e[2])

    for u_coord, v_coord, dist, direction in all_edges_unlimited:
        if find(u_coord) == find(v_coord):
            continue
        x1, y1 = u_coord
        x2, y2 = v_coord
        if x1 == x2:
            for y in range(min(y1, y2)+1, max(y1, y2)):
                        grid[x1][y].make_bridge(1, 1)
        else:
            for x in range(min(x1, x2)+1, max(x1, x2)):
                        grid[x][y1].make_bridge(1, 0)
            bridge_counts[(tuple(sorted((u_coord, v_coord))), direction)] = 1
            island_map[u_coord].i_count += 1
            island_map[v_coord].i_count += 1
            union(u_coord, v_coord)
        root0 = find(island_coords[0])
        if all(find(coord) == root0 for coord in island_coords):
            print(f"DEBUG: Граф связан после добавления дополнительных рёбер")
            break

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Принудительно соединяем все оставшиеся компоненты
    components = {}
    for coord in island_coords:
        root = find(coord)
        if root not in components:
            components[root] = []
        components[root].append(coord)
    
    if len(components) > 1:
        print(f"DEBUG: Принудительное соединение {len(components)} компонент")
        component_list = list(components.values())
        
        # Соединяем все компоненты с первой
        main_component = component_list[0]
        for i in range(1, len(component_list)):
            other_component = component_list[i]
            
            # Найти ближайшую пару островов между компонентами
            best_dist = float('inf')
            best_connection = None
            
            for u in main_component + component_list[0]:  # Добавляем уже соединенные
                for v in other_component:
                    if u[0] == v[0] or u[1] == v[1]:  # Только горизонтальные/вертикальные
                        dist = abs(u[0] - v[0]) + abs(u[1] - v[1])
                        # Проверяем, что путь свободен
                        path_clear = True
                        if u[0] == v[0]:  # Вертикальный
                            for y in range(min(u[1], v[1])+1, max(u[1], v[1])):
                                if grid[u[0]][y].n_type != 0:
                                    path_clear = False
                                    break
                        else:  # Горизонтальный
                            for x in range(min(u[0], v[0])+1, max(u[0], v[0])):
                                if grid[x][u[1]].n_type != 0:
                                    path_clear = False
                                    break
                        
                        if path_clear and dist < best_dist:
                            best_dist = dist
                            best_connection = (u, v)
            
            # Создаем соединение
            if best_connection:
                u, v = best_connection
                direction = 1 if u[0] == v[0] else 0
                
                # Добавляем мост
                if u[0] == v[0]:  # Вертикальный
                    for y in range(min(u[1], v[1])+1, max(u[1], v[1])):
                        grid[u[0]][y].make_bridge(1, 1)
                else:  # Горизонтальный
                    for x in range(min(u[0], v[0])+1, max(u[0], v[0])):
                        grid[x][u[1]].make_bridge(1, 0)
                
                # Обновляем структуры данных
                bridge_counts[(tuple(sorted((u, v))), direction)] = 1
                island_map[u].i_count += 1
                island_map[v].i_count += 1
                union(u, v)
                
                # Добавляем соединенную компоненту к главной
                main_component.extend(other_component)

    def path_clear_and_mark(x1, y1, x2, y2, orientation):
        if orientation == 1 and x1 == x2:
            for y in range(min(y1, y2)+1, max(y1, y2)):
                if grid[x1][y].n_type == 1:
                    return False
            for y in range(min(y1, y2)+1, max(y1, y2)):
                grid[x1][y].make_bridge(1, 1)
            return True
        if orientation == 0 and y1 == y2:
            for x in range(min(x1, x2)+1, max(x1, x2)):
                if grid[x][y1].n_type == 1:
                    return False
            for x in range(min(x1, x2)+1, max(x1, x2)):
                grid[x][y1].make_bridge(1, 0)
            return True
        return False

    zero_deg_islands = [coord for coord in island_coords if island_map[coord].i_count == 0]
    print(f"DEBUG: Островов с нулевой степенью: {len(zero_deg_islands)}")
    if zero_deg_islands:
        for u in zero_deg_islands:
            ux, uy = u
            candidates = []
            ty = uy - 1
            while ty >= 0 and grid[ux][ty].n_type == 0:
                ty -= 1
            if ty >= 0 and grid[ux][ty].n_type == 1:
                v = (ux, ty); candidates.append((abs(uy-ty), v, 1))
            ty = uy + 1
            while ty < h and grid[ux][ty].n_type == 0:
                ty += 1
            if ty < h and grid[ux][ty].n_type == 1:
                v = (ux, ty); candidates.append((abs(uy-ty), v, 1))
            tx = ux - 1
            while tx >= 0 and grid[tx][uy].n_type == 0:
                tx -= 1
            if tx >= 0 and grid[tx][uy].n_type == 1:
                v = (tx, uy); candidates.append((abs(ux-tx), v, 0))
            tx = ux + 1
            while tx < w and grid[tx][uy].n_type == 0:
                tx += 1
            if tx < w and grid[tx][uy].n_type == 1:
                v = (tx, uy); candidates.append((abs(ux-tx), v, 0))
            candidates.sort(key=lambda t: t[0])
            for _, v, orient in candidates:
                if island_map[u].i_count >= 8 or island_map[v].i_count >= 8:
                    continue
                (vx, vy) = v
                if path_clear_and_mark(ux, uy, vx, vy, orient):
                    island_map[u].i_count += 1
                    island_map[v].i_count += 1
                    key = (tuple(sorted((u, v))), orient)
                    if key not in bridge_counts:
                        bridge_counts[key] = 1
                    break

    # После построения MST и обеспечения min-degree=1, добавляем двойные мосты с обратной связью и ограничениями степеней
    def count_current_double_percentage():
        single_count = sum(1 for count in bridge_counts.values() if count == 1)
        double_count = sum(1 for count in bridge_counts.values() if count == 2)
        total_links = single_count + double_count
        return (double_count / total_links * 100) if total_links > 0 else 0

    # Сортируем ребра по приоритету для добавления двойных мостов
    edges_to_double = []
    for edge, count in bridge_counts.items():
        if count == 1:  # Только одинарные мосты
            u_coord, v_coord = edge[0]  # Получаем координаты из кортежа
            u_degree = island_map[u_coord].i_count
            v_degree = island_map[v_coord].i_count
            
            # Определяем приоритет: выше для тех, которые могут привести к степени 7 или 8
            priority = 0
            
            # Проверяем, не приведет ли удвоение к превышению степени 8
            if u_degree + 1 > 8 or v_degree + 1 > 8:
                continue  # Пропускаем этот мост, если он приведет к превышению степени 8
                
            # Приоритет для высоких степеней
            if u_degree >= 5 or v_degree >= 5:
                priority = max(u_degree, v_degree) + 50  # Приоритет по максимальной степени с бонусом
            else:
                priority = u_degree + v_degree  # Сумма степеней для остальных
                
            edges_to_double.append((edge, priority))
    
    # Сортируем по приоритету (высокие приоритеты в начале)
    edges_to_double.sort(key=lambda x: x[1], reverse=True)
    
    # Добавляем двойные мосты до достижения целевого процента
    tolerance = 15.0
    current_percentage = count_current_double_percentage()
    kfeedback = 2.0
    max_deg7 = max(0, int(round(target_islands * 0.04)))
    max_deg8 = max(0, int(round(target_islands * 0.02)))
    
    # Пытаемся добавить двойные мосты, пока не достигнем целевого процента или не закончатся возможности
    attempts = 0
    max_bridge_attempts = len(edges_to_double) * 3 # Увеличиваем количество попыток
    
    while edges_to_double and attempts < max_bridge_attempts:
        attempts += 1
        # Пересортируем по ошибке процента (обратная связь)
        error = target_double_bridge_percentage - current_percentage
        edges_to_double.sort(key=lambda x: x[1] + kfeedback * error, reverse=True)
        edge, priority = edges_to_double.pop(0)
        u_coord, v_coord = edge[0]  # Получаем координаты из кортежа
        
        # Проверяем, можно ли добавить двойной мост (не превысит ли степень 8)
        u_degree = island_map[u_coord].i_count
        v_degree = island_map[v_coord].i_count
        
        # Новый блок: не трогаем degree-7 центры
        if u_coord in deg7_centers or v_coord in deg7_centers:
            continue  # Не увеличиваем степень degree-7 центров
        
        if u_degree + 1 > 8 or v_degree + 1 > 8:
            continue # Пропускаем этот мост
        # Ограничения на количество степеней 7 и 8
        deg7_now = sum(1 for c in island_coords if island_map[c].i_count == 7)
        deg8_now = sum(1 for c in island_coords if island_map[c].i_count == 8)
        future_deg7 = deg7_now + (1 if (u_degree + 1 == 7) else 0) + (1 if (v_degree + 1 == 7) else 0)
        future_deg8 = deg8_now + (1 if (u_degree + 1 == 8) else 0) + (1 if (v_degree + 1 == 8) else 0)
        if future_deg7 > max_deg7 or future_deg8 > max_deg8:
            continue
        
        # Добавляем двойной мост
        bridge_counts[edge] = 2
        island_map[u_coord].i_count += 1
        island_map[v_coord].i_count += 1
        
        current_percentage = count_current_double_percentage()
        if abs(current_percentage - target_double_bridge_percentage) <= tolerance:
            break
            
    removal_attempts = 0
    max_removal_attempts = 50
    while current_percentage > target_double_bridge_percentage + tolerance and removal_attempts < max_removal_attempts:
        removal_attempts += 1
        double_edges_to_single = []
        for edge, count in bridge_counts.items():
            if count == 2:
                u_coord, v_coord = edge[0]
                u_degree = island_map[u_coord].i_count
                v_degree = island_map[v_coord].i_count
                
                priority = 0
                if u_coord in deg7_centers or v_coord in deg7_centers:
                    priority = -200
                elif u_degree == 7 or v_degree == 7:
                    priority = -100
                else:
                    priority = max(u_degree, v_degree)
                double_edges_to_single.append((edge, priority))
                  
        if not double_edges_to_single:
            break
            
        double_edges_to_single.sort(key=lambda x: x[1], reverse=True)
        
        edge_to_single, priority_to_remove = double_edges_to_single.pop(0)
        u_coord, v_coord = edge_to_single[0]
        
        if priority_to_remove <= -100:
            if not double_edges_to_single:
                break
            continue
        
        bridge_counts[edge_to_single] = 1
        island_map[u_coord].i_count -= 1
        island_map[v_coord].i_count -= 1
        
        current_percentage = count_current_double_percentage()
        if abs(current_percentage - target_double_bridge_percentage) <= tolerance:
            break

    # Подсчитываем начальное количество одинарных и двойных мостов после MST и дополнительных
    single_bridges_count_final = 0
    double_bridges_count_final = 0
    for count in bridge_counts.values():
         if count == 1:
             single_bridges_count_final += 1
         elif count == 2:
             double_bridges_count_final += 1

    total_bridges_after_additional = sum(bridge_counts.values())
    total_links = single_bridges_count_final + double_bridges_count_final
    
    # Вычисляем целевое количество двойных мостов на основе процента от общего количества связей
    target_double_links = round(total_links * (target_double_bridge_percentage / 100.0))

    # Подсчет и вывод статистики
    degree_counts_final = {}
    for coord in island_coords:
        degree = island_map[coord].i_count
        if degree == 0:
            continue  # Не считаем "острова" со степенью 0
        degree_counts_final[degree] = degree_counts_final.get(degree, 0) + 1
    
    print(f"DEBUG: Финальное распределение степеней: {degree_counts_final}")
    max_degree = max(degree_counts_final.keys()) if degree_counts_final else 0
    print(f"DEBUG: Максимальная степень: {max_degree}")
    print(f"DEBUG: Островов степени 7: {degree_counts_final.get(7, 0)} / целевая {target_degree_7}")
    print(f"DEBUG: Островов степени 8: {degree_counts_final.get(8, 0)} / целевая {target_degree_8}")
    
    # Проверим связность финального графа
    components = set()
    for coord in island_coords:
        components.add(find(coord))
    print(f"DEBUG: Количество компонент связности: {len(components)}")
    # Ранний отказ: если граф несвязный или есть острова степени 0 — просим перегенерацию
    if len(components) > 1 or any(island_map[c].i_count == 0 for c in island_coords):
        print("DEBUG: Early reject — несвязный граф или острова с нулевой степенью, повторяем попытку")
        return None, {}
    
    for (u_coord, v_coord), direction in bridge_counts.keys(): # bridge_counts хранит tuple(sorted(coords))
        if (tuple(sorted((u_coord, v_coord))), direction) in bridge_counts:
             count = bridge_counts[(tuple(sorted((u_coord, v_coord))), direction)]
             x1, y1 = u_coord
             x2, y2 = v_coord
             
             if x1 == x2: # Вертикальный мост
                 min_y, max_y = sorted((y1, y2))
                 for y in range(min_y + 1, max_y):
                     if is_in_grid(x1, y, len(grid), len(grid[0])):
                         grid[x1][y].make_bridge(count, 1) # Обновляем толщину (count) и ориентацию (1 для вертикального)
             else: # Горизонтальный мост
                 min_x, max_x = sorted((x1, x2))
                 for x in range(min_x + 1, max_x):
                      if is_in_grid(x, y1, len(grid), len(grid[0])):
                         grid[x][y1].make_bridge(count, 0) # Обновляем толщину (count) и ориентацию (0 для горизонтального)

    # Возвращаем сетку и статистику
    global current_puzzle_stats
    current_puzzle_stats = {
        'single_bridges': single_bridges_count_final,
        'double_bridges': double_bridges_count_final,
        'total_bridges': single_bridges_count_final + double_bridges_count_final * 2,
        'double_bridge_percentage': target_double_links / total_links * 100 if total_links > 0 else 0,
        'generation_time': time.time() - start_time  # Add generation time to stats
    }
    # Подсчет и вывод статистики
    degree_counts_final = {}
    for coord in island_coords:
        degree = island_map[coord].i_count
        if degree == 0:
            continue  # Не считаем "острова" со степенью 0
        degree_counts_final[degree] = degree_counts_final.get(degree, 0) + 1
    return grid, current_puzzle_stats


def main():
    generated_dir = os.path.join("examples", "generated")
    os.makedirs(generated_dir, exist_ok=True)

    # Собираем результаты генерации для сводки
    generation_results = defaultdict(list)

    # Определяем параметры для разных размеров сетки
    # (ширина, высота, количество островов, target_double_bridge_percentage, target_degree_7, target_degree_8, connectivity_factor_alpha)
    puzzle_configs = [
        (16, 16, 100, 25, 2, 2, 5),
        (16, 16, 100, 50, 4, 2, 10),
    ]

    # Генерируем по 2 задачи для каждой конфигурации
    num_puzzles = 30
    for w, h, islands, bridges, deg7, deg8, alpha in puzzle_configs:
        config_name = f"{w}x{h}_{islands}_{bridges}"
        config_dir = os.path.join(generated_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        print(f"\n{'='*40}")
        print(f"Генерация задач для конфигурации: {config_name} (Alpha={alpha}, Deg7={deg7}, Deg8={deg8})")
        print(f"{'='*40}")

        # --- Новый блок: читаем уже существующие имена файлов и статистику ---
        existing_filenames = set()
        stat_file_path = os.path.join(generated_dir, f"{config_name}_stats.txt")
        if os.path.exists(stat_file_path):
            with open(stat_file_path, 'r', encoding='utf-8') as stat_file:
                for line in stat_file:
                    m = re.match(r"^(Hs_.*?\.has)", line)
                    if m:
                        existing_filenames.add(m.group(1).strip())

        for i in range(num_puzzles):
            print(f"\nГенерация задачи {i+1}/{num_puzzles} для {config_name}...")
            max_attempts = 150
            attempt = 0
            grid = None
            total_generation_time = 0 
            
            while attempt < max_attempts:
                attempt += 1
                attempt_start_time = time.time()
                
                import io
                import contextlib
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                try:
                    grid, stats = generate_solvable_puzzle(w, h, islands,
                                                  target_double_bridge_percentage=bridges,
                                                  target_degree_7=deg7,
                                                  target_degree_8=deg8,
                                                  connectivity_factor_alpha=alpha)
                    generation_log = redirected_output.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                actual_islands = sum(1 for r in grid for node in r if node.n_type == 1 and node.i_count > 0) if grid else 0
                if actual_islands == islands and grid:
                    # Предварительная проверка тремя моделями (до сохранения файла)
                    try:
                        # Временно глушим логгирование внутренних решателей
                        logging.disable(logging.CRITICAL)
                        numeric_grid = [[0 for _ in range(w)] for _ in range(h)]
                        for ix in range(w):
                            for iy in range(h):
                                if grid[ix][iy].n_type == 1 and grid[ix][iy].i_count > 0:
                                    numeric_grid[iy][ix] = grid[ix][iy].i_count
                        from src.core.puzzle import Puzzle as CorePuzzle
                        puzzle_mem = CorePuzzle(w, h, numeric_grid)

                        # Original precheck
                        is_valid_orig_pre = False
                        orig_ms = 0.0
                        orig_iters = '-'
                        s1 = None
                        try:
                            om = OriginalModel(puzzle_mem)
                            oslv = OriginalSolver(om)
                            t0 = time.time()
                            s1, sol1, _it = oslv.solve_with_cuts()
                            orig_ms = (time.time() - t0) * 1000.0
                            orig_iters = _it
                            is_valid_orig_pre = (s1 == pywraplp.Solver.OPTIMAL or s1 == pywraplp.Solver.FEASIBLE) and is_solution_valid(puzzle_mem, sol1)
                        except Exception:
                            is_valid_orig_pre = False

                        # Flow precheck
                        is_valid_flow_pre = False
                        flow_ms = 0.0
                        s2 = None
                        try:
                            fm = FlowModel(puzzle_mem)
                            fslv = FlowSolver(fm)
                            t0 = time.time()
                            s2, sol2 = fslv.solve()
                            flow_ms = (time.time() - t0) * 1000.0
                            is_valid_flow_pre = (s2 == pywraplp.Solver.OPTIMAL or s2 == pywraplp.Solver.FEASIBLE) and is_solution_valid(puzzle_mem, sol2)
                        except Exception:
                            is_valid_flow_pre = False

                        # Modified precheck
                        is_valid_mod_pre = False
                        mod_ms = 0.0
                        mod_iters = '-'
                        try:
                            mslv = ModifiedIterativeILP(puzzle_mem)
                            t0 = time.time()
                            res = mslv.solve()
                            mod_ms = (time.time() - t0) * 1000.0
                            sol3 = res[0] if isinstance(res, tuple) else res
                            if isinstance(res, tuple) and len(res) >= 2:
                                mod_iters = res[1]
                            is_valid_mod_pre = sol3 is not None and is_solution_valid(puzzle_mem, sol3)
                        except Exception:
                            is_valid_mod_pre = False
                        finally:
                            logging.disable(logging.NOTSET)

                        # Подробный вывод по попытке (в требуемом формате и порядке: Modified, Original, Flow)
                        status_str_mod = f"OK in {mod_ms:.1f} ms" if is_valid_mod_pre else f"FAIL in {mod_ms:.1f} ms"
                        print(f"Precheck Modified: {status_str_mod}")
                        try:
                            status_str_orig = f"OK in {orig_ms:.1f} ms" if is_valid_orig_pre else f"FAIL (status={s1}) in {orig_ms:.1f} ms"
                        except Exception:
                            status_str_orig = f"FAIL in {orig_ms:.1f} ms"
                        print(f"Precheck Original: {status_str_orig}")
                        try:
                            status_str_flow = f"OK in {flow_ms:.1f} ms" if is_valid_flow_pre else f"FAIL (status={s2}) in {flow_ms:.1f} ms"
                        except Exception:
                            status_str_flow = f"FAIL in {flow_ms:.1f} ms"
                        print(f"Precheck Flow: {status_str_flow}")

                        if not (is_valid_orig_pre and is_valid_flow_pre and is_valid_mod_pre):
                            continue
                    except Exception as _e:
                        print(f"Precheck error: {_e}")
                        continue
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_degree_counts = {}
                    for r in grid:
                        for node in r:
                            if node.n_type == 1 and node.i_count > 0:
                                final_degree_counts[node.i_count] = final_degree_counts.get(node.i_count, 0) + 1
                    
                    base_filename = f"Hs_{w}_{h}_{islands}_{bridges}_7s{final_degree_counts.get(7,0)}_8s{final_degree_counts.get(8,0)}_a{alpha}_{current_time}"
                    idx = 1
                    while True:
                        filename = f"{base_filename}_{str(idx).zfill(3)}.has"
                        filepath = os.path.join(config_dir, filename)
                        if filename not in existing_filenames and not os.path.exists(filepath):
                            break
                        idx += 1
                    
                    save_grid(grid, filepath)
                    print(f"  Головоломка сохранена в файл: {filename}")
                    
                    # Add validation time to total generation time
                    validation_start_time = time.time()
                    try:
                        from src.core.puzzle import Puzzle
                        from src.core.ModifiedIterativeILP import ModifiedIterativeILP
                        from src.core.model import Model as OriginalModel
                        from src.core.solver import Solver as OriginalSolver
                        from src.core.model_flow import FlowModel
                        from src.core.solver_flow import FlowSolver
                        puzzle = Puzzle.from_file(filepath)

                        # Original Solver validation
                        is_valid_orig = False
                        try:
                            model = OriginalModel(puzzle)
                            solver = OriginalSolver(model)
                            status1, solution1, n_iter = solver.solve_with_cuts()
                            is_valid_orig = (status1 == pywraplp.Solver.OPTIMAL or status1 == pywraplp.Solver.FEASIBLE) and is_solution_valid(puzzle, solution1)
                        except Exception as e:
                            is_valid_orig = False
                        print(f"  Original Solver: {'OK' if is_valid_orig else 'FAIL'}")
                        
                        # Flow Solver validation
                        is_valid_flow = False
                        try:
                            model = FlowModel(puzzle)
                            solver = FlowSolver(model)
                            status2, solution2 = solver.solve()
                            is_valid_flow = (status2 == pywraplp.Solver.OPTIMAL or status2 == pywraplp.Solver.FEASIBLE) and is_solution_valid(puzzle, solution2)
                        except Exception as e:
                            is_valid_flow = False
                        print(f"  Flow Solver: {'OK' if is_valid_flow else 'FAIL'}")
                        
                        # Modified Solver validation
                        is_valid_modified = False
                        try:
                            solver = ModifiedIterativeILP(puzzle)
                            result = solver.solve()
                            solution3 = result[0] if isinstance(result, tuple) else result
                            is_valid_modified = is_solution_valid(puzzle, solution3)
                        except Exception as e:
                            is_valid_modified = False
                        print(f"  Modified Solver: {'OK' if is_valid_modified else 'FAIL'}")
                        
                        validation_time = time.time() - validation_start_time
                        total_generation_time = time.time() - attempt_start_time
                        
                        print(generation_log)

                        if not (is_valid_orig and is_valid_flow and is_valid_modified):
                            print("  Один из методов не нашел решение или нашел некорректное решение. Удаление файла.")
                            os.remove(filepath)
                            continue
                        else:
                            print(f"  Головоломка успешно проверена всеми методами")
                            puzzle_info = {
                                'filename': filename,
                                'degree_counts': final_degree_counts,
                                'double_bridge_percentage': stats['double_bridge_percentage'],
                                'single_bridges': stats['single_bridges'],
                                'double_bridges': stats['double_bridges'],
                                'total_bridges': stats['total_bridges'],
                                'attempts': attempt,
                                'generation_time': total_generation_time
                            }
                            generation_results[config_name].append(puzzle_info)
                            existing_filenames.add(filename)
                            break
                    except Exception as e:
                        print(f"  Ошибка при проверке решения: {e}. Удаление файла.")
                        os.remove(filepath)
                        continue
                elif attempt < max_attempts:
                    # Недостаточно островов — без лишнего вывода
                    pass
            if attempt >= max_attempts:
                print(f"  Не удалось сгенерировать головоломку с {islands} островами после {max_attempts} попыток")
                # Без дополнительной диагностики по запросу

    for config_name, results in generation_results.items():
        if not results:
            continue
        stat_file_path = os.path.join(generated_dir, f"{config_name}_stats.txt")
        
        with open(stat_file_path, 'w', encoding='utf-8') as stat_file:
            stat_file.write(f"{'Filename':40} | {'Islands':8} | {'Single':10} | {'Double':8} | {'Total':12} | {'Double %':10} | {'Att':3} | {'Time':8}\n")
            stat_file.write(f"{'-'*40} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*12} | {'-'*10} | {'-'*3} | {'-'*8}\n")
        
        existing_lines = set()
        if os.path.exists(stat_file_path):
            with open(stat_file_path, 'r', encoding='utf-8') as stat_file:
                for line in stat_file:
                    m = re.match(r"^(Hs_.*?\.has)", line)
                    if m:
                        existing_lines.add(m.group(1).strip())
        
        with open(stat_file_path, 'a', encoding='utf-8') as stat_file:
            for puzzle_info in results:
                filename = puzzle_info['filename']
                if filename in existing_lines:
                    continue
                num_islands = sum(puzzle_info['degree_counts'].values())
                single_bridges = puzzle_info.get('single_bridges', 0)
                double_bridges = puzzle_info.get('double_bridges', 0)
                total_bridges = puzzle_info.get('total_bridges', 0)
                double_percentage = puzzle_info['double_bridge_percentage']
                generation_time = puzzle_info.get('generation_time', 0)
                stat_file.write(f"{filename:40} | {num_islands:8} | {single_bridges:10} | {double_bridges:8} | {total_bridges:12} | {double_percentage:10.2f}% | {puzzle_info.get('attempts', '-'):3} | {generation_time:8.2f}с\n")

    print(f"\n{'='*40}")
    print("СВОДКА РЕЗУЛЬТАТОВ ГЕНЕРАЦИИ")
    print(f"{'='*40}")

    if not generation_results:
        print("Ни одной головоломки не было успешно сгенерировано.")
    else:
        for config_name, results in generation_results.items():
            print(f"\n--- Конфигурация: {config_name} ---")
            print(f"  Сгенерировано успешно: {len(results)}/{num_puzzles}")
            for puzzle_info in results:
                print(f"    Файл: {puzzle_info['filename']}")
                print(f"      Распределение степеней: {puzzle_info['degree_counts']}")
                print(f"      Процент двойных связей: {puzzle_info['double_bridge_percentage']:.2f}%")
                print(f"      Количество попыток генерации: {puzzle_info.get('attempts', '-')}")

    print(f"\n{'='*40}")
    print("Генерация завершена.")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
