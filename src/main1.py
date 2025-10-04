# main.py
import os
import sys
import time
import statistics
import argparse
import re

# Add both src and hashiwokakero to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hashiwokakero')))

from src.core.puzzle import Puzzle

# Импортируем FlowModel и FlowSolver
from src.core.model_flow import FlowModel
from src.core.solver_flow import FlowSolver

import networkx as nx
from ortools.linear_solver import pywraplp
from src.core.model import Model as OriginalModel
from src.core.solver import Solver as OriginalSolver
from src.core.ModifiedIterativeILP import ModifiedIterativeILP
from hashi.generator import generate_solvable_puzzle
from hashi.export import save_grid

from src.utils import is_solution_valid

import pandas as pd
import matplotlib.pyplot as plt

def _quick_feasible_grid(grid_nodes) -> bool:
    h = len(grid_nodes[0]) if grid_nodes else 0
    w = len(grid_nodes)
    if w == 0 or h == 0:
        return False
    deg_grid = [[0 for _ in range(h)] for _ in range(w)]
    islands = []
    for x in range(w):
        for y in range(h):
            node = grid_nodes[x][y]
            if getattr(node, 'n_type', 0) == 1:
                d = getattr(node, 'i_count', 0)
                deg_grid[x][y] = d
                islands.append((x, y, d))
    if not islands:
        return False
    sum_deg = sum(d for _, _, d in islands)
    n = len(islands)
    if sum_deg % 2 != 0 or sum_deg < 2 * (n - 1):
        return False
    idx_map = {(x, y): idx for idx, (x, y, _) in enumerate(islands)}
    adj = [[] for _ in range(n)]
    for idx, (x, y, d) in enumerate(islands):
        tx = x - 1
        while tx >= 0 and deg_grid[tx][y] == 0:
            tx -= 1
        if tx >= 0 and deg_grid[tx][y] > 0:
            adj[idx].append(idx_map[(tx, y)])
        # right
        tx = x + 1
        while tx < w and deg_grid[tx][y] == 0:
            tx += 1
        if tx < w and deg_grid[tx][y] > 0:
            adj[idx].append(idx_map[(tx, y)])
        # up
        ty = y - 1
        while ty >= 0 and deg_grid[x][ty] == 0:
            ty -= 1
        if ty >= 0 and deg_grid[x][ty] > 0:
            adj[idx].append(idx_map[(x, ty)])
        # down
        ty = y + 1
        while ty < h and deg_grid[x][ty] == 0:
            ty += 1
        if ty < h and deg_grid[x][ty] > 0:
            adj[idx].append(idx_map[(x, ty)])
        if d > 2 * len(adj[idx]):
            return False
    seen = set([0])
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != n:
        return False
    return True

def select_puzzle_folder():
    """Позволяет пользователю выбрать папку с головоломками."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hashi-generator', 'output'))
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    folders.append('hashi-generator/output')
    
    print("\nДоступные папки с головоломками:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = int(input("\nВыберите номер папки (1-{}): ".format(len(folders))))
            if 1 <= choice <= len(folders):
                if folders[choice-1] == 'hashi-generator/output':
                    return output_path
                return os.path.join(base_path, folders[choice-1])
            print("Неверный выбор. Попробуйте снова.")
        except ValueError:
            print("Пожалуйста, введите число.")

def calculate_metrics(puzzle, solution):
    """Вычисляет дополнительные метрики для решения."""
    if not solution:
        return {
            'bridge_count': 0,
            'island_connectivity': 0,
            'solution_density': 0,
            'max_bridges_per_island': 0
        }
    
    islands = puzzle.get_all_islands()
    total_islands = len(islands)
    total_possible_bridges = (total_islands * (total_islands - 1)) // 2
    bridge_count = sum(solution.values())
    connected_islands = set()
    for (i, j), bridges in solution.items():
        if bridges > 0:
            connected_islands.add(i)
            connected_islands.add(j)
    island_connectivity = len(connected_islands) / total_islands
    solution_density = bridge_count / total_possible_bridges if total_possible_bridges > 0 else 0
    
    island_bridges = {}
    for (i, j), bridges in solution.items():
        island_bridges[i] = island_bridges.get(i, 0) + bridges
        island_bridges[j] = island_bridges.get(j, 0) + bridges
    max_bridges_per_island = max(island_bridges.values()) if island_bridges else 0
    
    return {
        'bridge_count': bridge_count,
        'island_connectivity': island_connectivity,
        'solution_density': solution_density,
        'max_bridges_per_island': max_bridges_per_island
    }

def calculate_additional_graph_metrics(solution, puzzle):
    import networkx as nx
    from collections import Counter
    G = nx.Graph()
    for island in puzzle.get_all_islands():
        G.add_node(island['id'])
    degrees = Counter()
    double_bridges = 0
    single_bridges = 0
    for (i, j), bridges in solution.items():
        if bridges > 0:
            G.add_edge(i, j, weight=bridges)
            degrees[i] += bridges
            degrees[j] += bridges
            if bridges == 2:
                double_bridges += 1
            elif bridges == 1:
                single_bridges += 1
    metrics = {}
    if G.number_of_nodes() > 1 and nx.is_connected(G):
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        metrics['diameter'] = None
        metrics['avg_shortest_path'] = None
    metrics['density'] = nx.density(G)
    deg_list = list(degrees.values())
    metrics['avg_degree'] = sum(deg_list)/len(deg_list) if deg_list else 0
    metrics['max_degree'] = max(deg_list) if deg_list else 0
    metrics['min_degree'] = min(deg_list) if deg_list else 0
    for d in range(1, 9):
        metrics[f'degree_{d}_count'] = deg_list.count(d)
    metrics['double_bridge_percent'] = (double_bridges/(double_bridges+single_bridges)*100) if (double_bridges+single_bridges)>0 else 0
    try:
        cycles = nx.cycle_basis(G)
        metrics['num_cycles'] = len(cycles)
        metrics['min_cycle_len'] = min((len(c) for c in cycles), default=0)
        metrics['max_cycle_len'] = max((len(c) for c in cycles), default=0)
    except Exception:
        metrics['num_cycles'] = 0
        metrics['min_cycle_len'] = 0
        metrics['max_cycle_len'] = 0
    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
    except Exception:
        metrics['avg_clustering'] = 0
    # Симметричность (разница между max и min степенью)
    metrics['symmetry'] = 1 - (metrics['max_degree']-metrics['min_degree'])/8 if metrics['max_degree']>0 else 0
    return metrics

def run_test(puzzle_file, num_runs=5, ModelClass=OriginalModel, SolverClass=OriginalSolver):
    times = []
    algorithm_times = []
    solutions = []
    statuses = []
    best_solution = None
    best_time = float('inf')
    best_algorithm_time = float('inf')
    metrics = []
    iterations_list = []
    for i in range(num_runs):
        try:
            puzzle = Puzzle.from_file(puzzle_file)
        except FileNotFoundError:
            print(f"File not found: {puzzle_file}")
            return {'statuses': [], 'solutions': [], 'times': [], 'algorithm_times': [], 'best_solution': None, 'metrics': []}
        except Exception as e:
            print(f"Error loading puzzle from {puzzle_file}: {e}")
            return {'statuses': [], 'solutions': [], 'times': [], 'algorithm_times': [], 'best_solution': None, 'metrics': []}
        start_time = time.time()
        if ModelClass == ModifiedIterativeILP:
            solver = ModelClass(puzzle)
            algorithm_start_time = time.time()
            import io
            import sys
            from contextlib import redirect_stdout
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                solution_tuple = solver.solve()
                if isinstance(solution_tuple, tuple) and len(solution_tuple) == 3:
                    solution, n_iter, algo_time = solution_tuple
                else:
                    solution = solution_tuple[0] if isinstance(solution_tuple, tuple) else solution_tuple
                    n_iter = 1
                    algo_time = None
            output = captured_output.getvalue()
            algorithm_end_time = time.time()
            end_time = time.time()
            times.append(end_time - start_time)
            if algo_time is not None:
                algorithm_times.append(algo_time)
            else:
                algorithm_times.append(algorithm_end_time - algorithm_start_time)
            solutions.append(solution)
            status = pywraplp.Solver.OPTIMAL if solution else pywraplp.Solver.INFEASIBLE
            statuses.append(status)
            iterations_list.append(n_iter)
            if solution:
                m = calculate_metrics(puzzle, solution)
                m.update(calculate_additional_graph_metrics(solution, puzzle))
                m['iterations'] = n_iter
                metrics.append(m)
        else:
            model = ModelClass(puzzle)
            solver = SolverClass(model)
            algorithm_start_time = time.time()
            if SolverClass == OriginalSolver:
                status, solution, n_iter = solver.solve_with_cuts()
                if solution:
                    m = calculate_metrics(puzzle, solution)
                    m.update(calculate_additional_graph_metrics(solution, puzzle))
                else:
                    m = {}
                m['iterations'] = n_iter if n_iter is not None else 1
                metrics.append(m)
            elif SolverClass == FlowSolver:
                status, solution = solver.solve()
                n_iter = None
                if solution:
                    m = calculate_metrics(puzzle, solution)
                    m.update(calculate_additional_graph_metrics(solution, puzzle))
                else:
                    m = {}
                m['iterations'] = None
                metrics.append(m)
            else:
                print(f"Unknown SolverClass: {SolverClass}")
                return {'statuses': [], 'solutions': [], 'times': [], 'algorithm_times': [], 'best_solution': None, 'metrics': []}
            algorithm_end_time = time.time()
            end_time = time.time()
            times.append(end_time - start_time)
            algorithm_times.append(algorithm_end_time - algorithm_start_time)
            solutions.append(solution)
            statuses.append(status)
            iterations_list.append(n_iter)
        if len(metrics) < (i + 1):
            m = {'iterations': n_iter if 'n_iter' in locals() and n_iter is not None else None}
            metrics.append(m)
    return {
        'statuses': statuses,
        'solutions': solutions,
        'times': times,
        'algorithm_times': algorithm_times,
        'best_solution': best_solution,
        'metrics': metrics,
        'iterations': iterations_list
    }

def generate_latex_table(results):
    """Генерирует LaTeX таблицу с результатами."""
    if not results:
        return "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\nNo results available\n\\hline\n\\end{tabular}"

    # Группируем результаты по размеру сетки
    grid_sizes = {}
    for puzzle_name, puzzle_results in results.items():
        # Извлекаем размер сетки из имени файла
        # Формат имени: Hs_16_100_25_00_001.has
        size = int(puzzle_name.split('_')[1])
        if size not in grid_sizes:
            grid_sizes[size] = []
        grid_sizes[size].append(puzzle_results)

    # Создаем строки таблицы
    rows = []
    for size in sorted(grid_sizes.keys()):
        metrics = grid_sizes[size]
        if not metrics:  # Пропускаем пустые метрики
            continue
            
        # Вычисляем средние значения для каждой модели
        avg_times = {}
        for model_name in ['Original', 'Flow', 'Modified']:
            times = [m[model_name]['avg_time'] for m in metrics if model_name in m]
            if times:
                avg_times[model_name] = statistics.mean(times)
            else:
                avg_times[model_name] = 0

        # Находим лучшее время
        best_time = min(avg_times.values())
        
        row = f"{size} & {best_time:.3f} & {avg_times['Original']:.3f} & {avg_times['Flow']:.3f} & {avg_times['Modified']:.3f} \\\\" 
        rows.append(row)

    # Формируем таблицу
    table = "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
    table += "Grid Size & Best Time & Original & Flow & Modified \\\n\\hline\n"
    table += "\n".join(rows)
    table += "\n\\hline\n\\end{tabular}"
    
    return table

def visualize_solution(puzzle_file, solution, title="Hashiwokakakero Solution", output_dir=None):
    """
    Визуализирует решение головоломки.
    """
    puzzle = Puzzle.from_file(puzzle_file)
    
    # Создаем граф для визуализации
    G = nx.Graph()
    
    # Добавляем узлы (острова)
    island_positions = {}
    island_labels = {}
    islands_data = puzzle.get_all_islands()
    for island in islands_data:
        island_id = island['id']
        island_x = island['x'] # Горизонтальная позиция (колонка)
        island_y = island['y'] # Вертикальная позиция (строка)
        island_degree = island['degree']
        
        G.add_node(island_id, pos=(island_x, -island_y)) # Используем x, -y для позиционирования
        island_positions[island_id] = (island_x, -island_y)
        island_labels[island_id] = str(island_degree)

    # Рисуем граф
    fig, ax = plt.subplots(figsize=(10, 10))

    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos=island_positions, node_size=800, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos=island_positions, labels=island_labels, font_size=10, font_weight='bold', ax=ax)

    # Рисуем мосты
    for (u_id, v_id), bridges in solution.items():
        if bridges > 0:
            u_pos = island_positions[u_id]
            v_pos = island_positions[v_id]

        if bridges == 1:
                # Рисуем один мост
                ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], color='gray', linewidth=2, zorder=-1)
        elif bridges == 2:
                # Рисуем два моста со небольшим смещением
                if u_pos[0] == v_pos[0]: # Вертикальный мост
                    offset = 0.05 * (v_pos[1] - u_pos[1]) / abs(v_pos[1] - u_pos[1]) # Смещение по X
                    ax.plot([u_pos[0] - offset, v_pos[0] - offset], [u_pos[1], v_pos[1]], color='gray', linewidth=2, zorder=-1)
                    ax.plot([u_pos[0] + offset, v_pos[0] + offset], [u_pos[1], v_pos[1]], color='gray', linewidth=2, zorder=-1)
                else: # Горизонтальный мост
                    offset = 0.05 * (v_pos[0] - u_pos[0]) / abs(v_pos[0] - u_pos[0]) # Смещение по Y
                    ax.plot([u_pos[0], v_pos[0]], [u_pos[1] + offset, v_pos[1] + offset], color='gray', linewidth=2, zorder=-1)
                    ax.plot([u_pos[0], v_pos[0]], [u_pos[1] - offset, v_pos[1] - offset], color='gray', linewidth=2, zorder=-1)
    
    plt.title(title)
    plt.axis('equal') # Равный масштаб по осям
    plt.axis('off')
    
    # Сохраняем изображение в папку результатов
    if output_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'generator_results')
    else:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True) # Убедимся, что директория существует
    
    puzzle_name = os.path.basename(puzzle_file).replace('.has', '').replace('.txt', '')
    image_path = os.path.join(results_dir, f'{puzzle_name}_solution.png')
    plt.savefig(image_path)
    print(f"Решение сохранено как изображение: {image_path}")
    plt.close(fig)

def get_best_model_for_puzzle(puzzle_results):
    """Определяет лучшую модель для конкретной головоломки."""
    best_model = None
    best_time = float('inf')
    
    for model_name, stats in puzzle_results.items():
        if stats['avg_time'] < best_time:
            best_time = stats['avg_time']
            best_model = model_name
    
    return best_model, best_time

def print_summary(results, folder_name):
    """Выводит общую сводку результатов тестирования и сохраняет её в файл."""
    summary = []
    summary.append("="*80)
    summary.append("ОБЩАЯ СВОДКА РЕЗУЛЬТАТОВ")
    summary.append("="*80)
    
    # Счетчики для каждой модели
    model_wins = {'Original': 0, 'Flow': 0, 'Modified': 0}
    total_puzzles = len(results)
    
    summary.append("\nЛучшие результаты по головоломкам:")
    summary.append("-"*80)
    
    for puzzle_name, puzzle_results in results.items():
        best_model, best_time = get_best_model_for_puzzle(puzzle_results)
        model_wins[best_model] += 1
        
        summary.append(f"\n{puzzle_name}:")
        summary.append(f"  Лучшая модель: {best_model} (время: {best_time:.4f} сек)")
        summary.append("  Результаты всех моделей:")
        for model_name, stats in puzzle_results.items():
            summary.append(f"    {model_name}: {stats['avg_time']:.4f} сек")
    
    summary.append("\n" + "="*80)
    summary.append("ИТОГОВАЯ СТАТИСТИКА")
    summary.append("="*80)
    summary.append(f"Всего протестировано головоломок: {total_puzzles}")
    summary.append("\nКоличество побед каждой модели:")
    for model, wins in model_wins.items():
        percentage = (wins / total_puzzles) * 100
        summary.append(f"  {model}: {wins} побед ({percentage:.1f}%)")
    
    # Выводим в консоль
    print("\n".join(summary))
    
    # Сохраняем в файл
    summary_file = f'summary_{folder_name}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary))
    
    return summary_file

def show_menu():
    """Показывает главное меню программы и возвращает выбор (1-3)."""
    print("\nГлавное меню:")
    print("1. Создать новые головоломки")
    print("2. Протестировать существующие головоломки")
    print("3. Выйти")
    return _ask_int("\nВыберите действие (1-3)", 1, 3, 1)

def _ask_int(prompt: str, min_v: int, max_v: int, default: int = None) -> int:
    """Запрашивает у пользователя целое число с проверкой диапазона."""
    while True:
        s = input(f"{prompt} " + (f"[по умолчанию {default}]: " if default is not None else ": "))
        s = s.strip()
        if not s and default is not None:
            return default
        try:
            val = int(s)
            if val < min_v or val > max_v:
                print(f"Ввод вне диапазона [{min_v}..{max_v}]. Попробуйте снова.")
                continue
            return val
        except ValueError:
            print("Пожалуйста, введите целое число.")


def _solve_with_original(puzzle_path: str) -> bool:
    try:
        puzzle = Puzzle.from_file(puzzle_path)
        model = OriginalModel(puzzle)
        solver = OriginalSolver(model)
        status, solution, _ = solver.solve_with_cuts()
        return bool(solution) and is_solution_valid(puzzle, solution)
    except Exception:
        return False

def _solve_with_flow(puzzle_path: str) -> bool:
    try:
        puzzle = Puzzle.from_file(puzzle_path)
        model = FlowModel(puzzle)
        solver = FlowSolver(model)
        status, solution = solver.solve()
        return bool(solution) and is_solution_valid(puzzle, solution)
    except Exception:
        return False

def _solve_with_modified(puzzle_path: str) -> bool:
    try:
        puzzle = Puzzle.from_file(puzzle_path)
        solver = ModifiedIterativeILP(puzzle)
        result = solver.solve()
        solution = result[0] if isinstance(result, tuple) else result
        return bool(solution) and is_solution_valid(puzzle, solution)
    except Exception:
        return False

def _is_solvable_by_any(puzzle_path: str) -> bool:
    # Достаточно, чтобы хотя бы один из решателей нашел корректное решение
    return _solve_with_original(puzzle_path) or _solve_with_flow(puzzle_path) or _solve_with_modified(puzzle_path)

def _is_solvable_by_all(puzzle_path: str) -> bool:
    # Требуем, чтобы все три решателя нашли корректное решение
    return _solve_with_original(puzzle_path) and _solve_with_flow(puzzle_path) and _solve_with_modified(puzzle_path)

def generate_puzzles():
    """Интерактивно генерирует новые головоломки и сохраняет решения в generator_results."""
    # Путь к директории для результатов
    results_root = os.path.join(os.path.dirname(__file__), '..', 'examples', 'generator_results')
    os.makedirs(results_root, exist_ok=True)

    print("\nРежим генерации:")
    print("1. Предустановленные группы (несколько конфигураций)")
    print("2. Свои параметры")
    print("3. Назад")

    choice = _ask_int("Выберите режим (1-3)", 1, 3, 1)
    if choice == 3:
        return True

    configs = []  # список кортежей (w, h, islands, percent)

    if choice == 1:
        print("\nДоступные группы:")
        print("1) 7x7, 12 островов, проценты: 25, 50, 75")
        print("2) 10x10, 24 острова, проценты: 25, 50, 75")
        print("3) 16x16, 100 островов, проценты: 25, 50")
        g = _ask_int("Выберите группу (1-3)", 1, 3, 3)
        num_puzzles = _ask_int("Сколько головоломок генерировать на конфигурацию (1-30)", 1, 30, 5)
        if g == 1:
            base = (7, 7, 12)
            percents = [25, 50, 75]
        elif g == 2:
            base = (10, 10, 24)
            percents = [25, 50, 75]
        else:
            base = (16, 16, 100)
            percents = [25, 50]
        for p in percents:
            for _ in range(num_puzzles):
                configs.append((base[0], base[1], base[2], p))
    else:
        # Свои параметры с ограничениями
        print("\nВведите параметры (разумные ограничения будут применены):")
        w = _ask_int("Ширина сетки w (7..40)", 7, 40, 16)
        h = _ask_int("Высота сетки h (7..40)", 7, 40, 16)
        max_islands = min(w * h // 2, 250)
        islands = _ask_int(f"Количество островов n (5..{max_islands})", 5, max_islands, min(100, max_islands))
        percent = _ask_int("Процент двойных мостов (0..100)", 0, 100, 25)
        num_puzzles = _ask_int("Сколько головоломок генерировать (1..100)", 1, 100, 5)
        for _ in range(num_puzzles):
            configs.append((w, h, islands, percent))

    print("\nЗапуск генератора...")
    try:
        for idx, (w, h, n_islands, percent) in enumerate(configs, 1):
            print(f"\n[{idx}/{len(configs)}] Генерация пазла {w}x{h} с {n_islands} островами, двойные мосты ~{percent}%...")

            # Имя каталога и файла в стиле Hs_{w}_{islands}_{percent}_00
            puzzle_name = f"Hs_{w}_{n_islands}_{percent:02d}_00"
            puzzle_dir = os.path.join(results_root, puzzle_name)
            os.makedirs(puzzle_dir, exist_ok=True)

            attempts = 0
            saved = False
            while not saved:
                attempts += 1
                try:
                    generated = generate_solvable_puzzle(
                        w, h, n_islands,
                        target_double_bridge_percentage=percent
                    )
                    grid = generated[0] if isinstance(generated, tuple) else generated
                except Exception as gen_err:
                    print(f"  Попытка {attempts}: ошибка генерации: {gen_err}")
                    continue

                # Быстрый тест осуществимости до записи на диск
                try:
                    if not _quick_feasible_grid(grid):
                        # Несовместимые степени/соседство — генерируем заново
                        continue
                except Exception:
                    continue

                existing_files = [f for f in os.listdir(puzzle_dir) if f.endswith('.has')]
                next_index = len(existing_files) + 1
                puzzle_path = os.path.join(puzzle_dir, f"{puzzle_name}_{str(next_index).zfill(3)}.has")

                try:
                    save_grid(grid, puzzle_path)
                except Exception as save_err:
                    print(f"  Попытка {attempts}: ошибка сохранения: {save_err}")
                    continue

                # Проверяем тремя решателями и печатаем Precheck строки (Modified, Original, Flow)
                try:
                    import logging
                    logging.disable(logging.CRITICAL)
                    # Modified
                    t0 = time.time()
                    try:
                        pz = Puzzle.from_file(puzzle_path)
                        mslv = ModifiedIterativeILP(pz)
                        res = mslv.solve()
                        sol3 = res[0] if isinstance(res, tuple) else res
                        ok_mod = sol3 is not None and is_solution_valid(pz, sol3)
                    except Exception:
                        ok_mod = False
                    mod_ms = (time.time() - t0) * 1000.0
                    print(f"Precheck Modified: {'OK' if ok_mod else 'FAIL'} in {mod_ms:.1f} ms")

                    # Original
                    t0 = time.time()
                    try:
                        om = OriginalModel(pz)
                        oslv = OriginalSolver(om)
                        s1, sol1, _it = oslv.solve_with_cuts()
                        ok_orig = (s1 == pywraplp.Solver.OPTIMAL or s1 == pywraplp.Solver.FEASIBLE) and is_solution_valid(pz, sol1)
                    except Exception:
                        s1 = None
                        ok_orig = False
                    orig_ms = (time.time() - t0) * 1000.0
                    print(f"Precheck Original: {'OK' if ok_orig else f'FAIL (status={s1})'} in {orig_ms:.1f} ms")

                    # Flow
                    t0 = time.time()
                    try:
                        fm = FlowModel(pz)
                        fslv = FlowSolver(fm)
                        s2, sol2 = fslv.solve()
                        ok_flow = (s2 == pywraplp.Solver.OPTIMAL or s2 == pywraplp.Solver.FEASIBLE) and is_solution_valid(pz, sol2)
                    except Exception:
                        s2 = None
                        ok_flow = False
                    flow_ms = (time.time() - t0) * 1000.0
                    print(f"Precheck Flow: {'OK' if ok_flow else f'FAIL (status={s2})'} in {flow_ms:.1f} ms")
                finally:
                    logging.disable(logging.NOTSET)

                if ok_mod and ok_orig and ok_flow:
                    print(f"  Успех на попытке {attempts}: пазл сохранен в {puzzle_path}")
                    # Дополнительно сохраним картинку решения Original, если получится
                    try:
                        if _solve_with_original(puzzle_path):
                            # Перерешаем одной попыткой для картинки
                            puzzle = Puzzle.from_file(puzzle_path)
                            model = OriginalModel(puzzle)
                            solver = OriginalSolver(model)
                            status, solution, _ = solver.solve_with_cuts()
                            if solution:
                                visualize_solution(puzzle_path, solution, title=f"{os.path.basename(puzzle_path)} (Original)", output_dir=puzzle_dir)
                    except Exception as vis_err:
                        print(f"  Не удалось сохранить изображение решения: {vis_err}")
                    saved = True
                else:
                    # Удаляем не решаемый файл и пробуем снова
                    try:
                        os.remove(puzzle_path)
                        # Ничего дополнительно не выводим — Precheck строки уже выведены
                    except Exception:
                        pass

            if not saved:
                print(f"  Не удалось получить решаемую головоломку для конфигурации {puzzle_name}")

        print("\nВсе пазлы сгенерированы успешно!")
        return True
    except Exception as e:
        print(f"\nОшибка при генерации пазлов: {e}")
        return False

def aggregate_and_save_results(folder_name, all_results):
    import pandas as pd
    import os
    print(f"[DEBUG] aggregate_and_save_results called for folder: {folder_name}")
    
    # Разделяем результаты по подпапкам
    subfolder_results = {}
    for puzzle_file, model_results in all_results.items():
        # Получаем имя подпапки из пути к файлу
        subfolder = os.path.basename(os.path.dirname(puzzle_file))
        if subfolder not in subfolder_results:
            subfolder_results[subfolder] = {}
        subfolder_results[subfolder][puzzle_file] = model_results
    
    # Обрабатываем каждую подпапку отдельно
    for subfolder, results in subfolder_results.items():
        rows = []
        for puzzle_file, model_results in results.items():
            for model_name, stats in model_results.items():
                for i, t in enumerate(stats['times']):
                    row = {
                        'Puzzle': os.path.basename(puzzle_file),
                        'Model': model_name,
                        'Run': i+1,
                        'Status': stats['statuses'][i] if i < len(stats['statuses']) else None,
                        'Time': t,
                        'AlgoTime': stats['algorithm_times'][i] if i < len(stats['algorithm_times']) else None,
                        'Iterations': stats['iterations'][i] if 'iterations' in stats and i < len(stats['iterations']) else None,
                    }
                    # Добавляем все метрики, если есть
                    if stats['metrics'] and i < len(stats['metrics']):
                        m = stats['metrics'][i].copy()
                        # Удаляем лишние метрики
                        for k in ['diameter','avg_degree','max_degree','min_degree','dangling_islands_percent']:
                            if k in m:
                                del m[k]
                        row.update(m)
                    rows.append(row)
        
        if not rows:
            continue
            
        df = pd.DataFrame(rows)
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'generated'))
        os.makedirs(output_dir, exist_ok=True)
        
        # Создаем подпапку для результатов
        subfolder_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(subfolder_output_dir, exist_ok=True)
        
        csv_path = os.path.join(subfolder_output_dir, f"results_{subfolder}.csv")
        summary_path = os.path.join(subfolder_output_dir, f"summary_{subfolder}.csv")
        txt_path = os.path.join(subfolder_output_dir, f"summary_{subfolder}.txt")
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Агрегируем по модели
        summary = []
        for model in df['Model'].unique():
            d = df[df['Model'] == model]
            s = {
                'Model': model,
                'Count': len(d),
                'Successes': d['Status'].notnull().sum(),
                'SuccessRate': d['Status'].notnull().mean(),
                'AvgTime': d['Time'].mean(),
                'MedianTime': d['Time'].median(),
                'MinTime': d['Time'].min(),
                'MaxTime': d['Time'].max(),
                'StdTime': d['Time'].std(),
                'AvgAlgoTime': d['AlgoTime'].mean(),
                'AvgIterations': d['Iterations'].mean() if 'Iterations' in d else None,
            }
            # Добавляем средние по всем новым метрикам, кроме удалённых
            for col in df.columns:
                if col not in s and col not in ['Puzzle','Model','Run','Status','Time','AlgoTime','Iterations','diameter','avg_degree','max_degree','min_degree','dangling_islands_percent'] and d[col].dtype != 'O':
                    s[f'Avg_{col}'] = d[col].mean()
            summary.append(s)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        
        # Красивый вывод в консоль и сохранение в txt
        txt_lines = []
        txt_lines.append(f"{'='*80}")
        txt_lines.append(f"ОБЩАЯ СТАТИСТИКА ПО ПАПКЕ: {subfolder}")
        txt_lines.append(f"{'='*80}")
        best_model = None
        best_time = float('inf')
        best_success = 0
        for idx, row in summary_df.iterrows():
            txt_lines.append(f"Модель: {row['Model']}")
            txt_lines.append(f"  Количество запусков: {row['Count']}")
            txt_lines.append(f"  Успешных решений: {row['Successes']} ({row['SuccessRate']*100:.1f}%)")
            txt_lines.append(f"  Среднее время: {row['AvgTime']:.4f} сек")
            txt_lines.append(f"  Медиана времени: {row['MedianTime']:.4f} сек")
            txt_lines.append(f"  Мин/Макс время: {row['MinTime']:.4f} / {row['MaxTime']:.4f} сек")
            txt_lines.append(f"  Ст. отклонение времени: {row['StdTime']:.4f}")
            avg_iter = row['AvgIterations']
            if row['Model'].lower().startswith('flow'):
                txt_lines.append(f"  Среднее число итераций: не рассчитывается")
            else:
                txt_lines.append(f"  Среднее число итераций: {avg_iter:.2f}" if pd.notnull(avg_iter) else "  Среднее число итераций: -")
            txt_lines.append(f"  Среднее время алгоритма: {row['AvgAlgoTime']:.4f} сек")
            for k, v in row.items():
                if k.startswith('Avg_') and v is not None:
                    txt_lines.append(f"  {k}: {v:.4f}")
            txt_lines.append('-'*60)
            # Для итогового вывода
            if row['SuccessRate'] == 1.0 and row['AvgAlgoTime'] < best_time:
                best_time = row['AvgAlgoTime']
                best_model = row['Model']
            elif row['SuccessRate'] > best_success:
                best_success = row['SuccessRate']
        
        # Итоговый вывод
        txt_lines.append(f"{'='*80}")
        if best_model:
            txt_lines.append(f"ЛУЧШАЯ МОДЕЛЬ ПО СРЕДНЕМУ ВРЕМЕНИ АЛГОРИТМА (100% успешных решений): {best_model} ({best_time:.4f} сек)")
        else:
            # Если нет моделей с 100% успехом, выбрать по максимальному SuccessRate
            best_row = summary_df.loc[summary_df['SuccessRate'].idxmax()]
            txt_lines.append(f"ЛУЧШАЯ МОДЕЛЬ ПО ПРОЦЕНТУ УСПЕШНЫХ РЕШЕНИЙ: {best_row['Model']} ({best_row['SuccessRate']*100:.1f}%)")
        txt_lines.append(f"{'='*80}")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_lines))
        
        print(f"\n{'='*80}\nОБЩАЯ СТАТИСТИКА ПО ПАПКЕ: {subfolder}\n{'='*80}")
        print('\n'.join(txt_lines))
        print(f"\nПодробные результаты сохранены в {csv_path}\nСводная статистика сохранена в {summary_path}\nТекстовая сводка сохранена в {txt_path}\n")

    # Create combined summary file
    combined_summary_path = os.path.join("examples", "generated", "combined_summary.txt")
    combined_lines = []
    combined_lines.append(f"{'='*80}")
    combined_lines.append("СВОДНАЯ СТАТИСТИКА ПО ВСЕМ КОНФИГУРАЦИЯМ")
    combined_lines.append(f"{'='*80}\n")
    
    for subfolder in sorted(os.listdir(os.path.join("examples", "generated"))):
        if not os.path.isdir(os.path.join("examples", "generated", subfolder)):
            continue
            
        summary_path = os.path.join("examples", "generated", subfolder, f"summary_{subfolder}.txt")
        if not os.path.exists(summary_path):
            continue
            
        combined_lines.append(f"\nКонфигурация: {subfolder}")
        combined_lines.append(f"{'-'*40}")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract metrics for each model
        models = ['Original', 'Flow', 'Modified']
        for model in models:
            model_section = content.split(f"Модель: {model}")[1].split("------------------------------------------------------------")[0]
            
            # Extract required metrics
            double_bridge_percent = re.search(r"Avg_double_bridge_percent: ([\d.]+)", model_section)
            avg_algo_time = re.search(r"Среднее время алгоритма: ([\d.]+) сек", model_section)
            avg_time = re.search(r"Среднее время: ([\d.]+) сек", model_section)
            iterations = re.search(r"Avg_iterations: ([\d.]+)", model_section)
            min_max_time = re.search(r"Мин/Макс время: ([\d.]+) / ([\d.]+) сек", model_section)
            std_time = re.search(r"Ст. отклонение времени: ([\d.]+)", model_section)
            
            combined_lines.append(f"\n{model}:")
            if double_bridge_percent:
                combined_lines.append(f"  Процент двойных мостов: {double_bridge_percent.group(1)}%")
            if avg_algo_time:
                combined_lines.append(f"  Среднее время алгоритма: {avg_algo_time.group(1)} сек")
            if avg_time:
                combined_lines.append(f"  Среднее время: {avg_time.group(1)} сек")
            if iterations:
                combined_lines.append(f"  Среднее число итераций: {iterations.group(1)}")
            if min_max_time:
                combined_lines.append(f"  Мин/Макс время: {min_max_time.group(1)} / {min_max_time.group(2)} сек")
            if std_time:
                combined_lines.append(f"  Ст. отклонение времени: {std_time.group(1)}")
        
        combined_lines.append(f"\n{'-'*40}")
    
    # Save combined summary
    with open(combined_summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_lines))
    
    print(f"\nСводная статистика по всем конфигурациям сохранена в {combined_summary_path}")

def main():
    # Добавляем парсер аргументов
    parser = argparse.ArgumentParser(description='Hashiwokakakero puzzle generator and tester')
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode')
    parser.add_argument('--test-folders', nargs='+', help='List of folders to test in automatic mode (command line)')
    parser.add_argument('--test-folders-file', type=str, help='Path to a file containing a list of folders to test (one per line)')
    args = parser.parse_args()

    test_folders = []
    if args.test_folders_file:
        try:
            with open(args.test_folders_file, 'r', encoding='utf-8') as f:
                test_folders = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Ошибка: Файл со списком папок не найден: {args.test_folders_file}")
            return
    elif args.test_folders:
        test_folders = args.test_folders

    # Если запущен в автоматическом режиме с указанием папок (через аргумент или файл)
    if args.auto and test_folders:
        all_folders_results = {}

        for folder_path in test_folders:
            folder_name = os.path.basename(folder_path)
            print(f"\nОбработка папки: {folder_name}")

            # Ищем все .has и .txt файлы
            puzzle_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.has', '.txt')):
                        puzzle_files.append(os.path.join(root, file))

            if not puzzle_files:
                print(f"В папке {folder_name} не найдено головоломок для тестирования.")
                continue

            # Выбираем только первые 5 головоломок
            puzzles_to_test = puzzle_files[:5]
            print(f"Тестирование {len(puzzles_to_test)} головоломок из папки {folder_name}...")

            folder_results = {}

            # Тестируем все три модели для выбранных головоломок
            models = [
                ('Original', OriginalModel, OriginalSolver),
                ('Flow', FlowModel, FlowSolver),
                ('Modified', ModifiedIterativeILP, None)
            ]

            for model_name, ModelClass, SolverClass in models:
                model_times = []
                model_algo_times = []
                model_metrics = []
                model_statuses = []

                print(f"  Тестирование модели: {model_name}")

                for puzzle_file in puzzles_to_test:
                    stats = run_test(puzzle_file, num_runs=1, ModelClass=ModelClass, SolverClass=SolverClass) # Уменьшил num_runs до 1 для скорости

                    if stats['times']:
                        model_times.extend(stats['times'])
                        model_algo_times.extend(stats['algorithm_times'])
                        model_metrics.extend(stats['metrics'])
                        model_statuses.extend(stats['statuses'])
                
                if model_times:
                    folder_results[puzzle_file] = {
                        'times': model_times,
                        'algorithm_times': model_algo_times,
                        'metrics': model_metrics,
                        'statuses': model_statuses
                    }
            
            all_folders_results[folder_name] = folder_results
            # После тестирования папки — агрегируем и сохраняем
            aggregate_and_save_results(folder_name, folder_results)
        
        print("\nТестирование завершено.")
        return
    
    # Обычный интерактивный режим
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Создание новых головоломок
            success = generate_puzzles()
            if success:
                print("\nГоловоломки успешно созданы!")
            else:
                print("\nПроизошла ошибка при создании головоломок.")
        
        elif choice == 2:
            # Тестирование существующих головоломок
            folder_path = select_puzzle_folder()
            folder_name = os.path.basename(folder_path)
            # Ищем все .has и .txt файлы, включая подпапки
            puzzle_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.has', '.txt')):
                        puzzle_files.append(os.path.join(root, file))
            if not puzzle_files:
                print(f"\nВ папке {folder_name} не найдено головоломок для тестирования.")
                continue
            results = {}
            print(f"\nРезультаты тестирования алгоритмов для папки: {folder_name}")
            print("=" * 80)
            models = [
                ('Original', OriginalModel, OriginalSolver),
                ('Flow', FlowModel, FlowSolver),
                ('Modified', ModifiedIterativeILP, None)
            ]
            for model_name, ModelClass, SolverClass in models:
                print(f"  Тестирование модели: {model_name}")
                for puzzle_file in puzzle_files:
                    stats = run_test(puzzle_file, num_runs=1, ModelClass=ModelClass, SolverClass=SolverClass)
                    if stats['times']:
                        if puzzle_file not in results:
                            results[puzzle_file] = {}
                        results[puzzle_file][model_name] = {
                            'times': stats['times'],
                            'algorithm_times': stats['algorithm_times'],
                            'metrics': stats['metrics'],
                            'statuses': stats['statuses']
                        }
                        # --- ВИЗУАЛИЗАЦИЯ РЕШЕНИЯ ---
                        # Берем первое решение, если оно есть
                        if stats['solutions'] and stats['solutions'][0]:
                            solution = stats['solutions'][0]
                            # Формируем путь сохранения изображения
                            puzzle_name = os.path.basename(puzzle_file).replace('.has', '').replace('.txt', '')
                            results_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'generator_results')
                            os.makedirs(results_dir, exist_ok=True)
                            image_path = os.path.join(results_dir, f'{puzzle_name}_solution_{model_name}.png')
                            if not os.path.exists(image_path):
                                visualize_solution(puzzle_file, solution, title=f"{puzzle_name} ({model_name})")
            # После вывода — сохраняем результаты
            if results:
                aggregate_and_save_results(folder_name, results)
            print("\nТестирование завершено.")
        
        elif choice == 3:
            print("\nДо свидания!")
            break

# Нужна новая функция для запуска тестирования по списку файлов в режиме интерактивного меню
def run_test_folder(puzzle_files, num_runs=2, ModelClass=OriginalModel, SolverClass=OriginalSolver):
    """Запускает несколько прогонов решения головоломок из списка файлов."""
    all_times = []
    all_algorithm_times = []
    # all_metrics = [] # Пока не собираем метрики в этой функции
    
    for puzzle_file in puzzle_files:
         # Запускаем run_test для каждого файла
         stats = run_test(puzzle_file, num_runs=num_runs, ModelClass=ModelClass, SolverClass=SolverClass)
         if stats['times']:
              all_times.extend(stats['times'])
              all_algorithm_times.extend(stats['algorithm_times'])
              # if stats['metrics']:
              #      all_metrics.extend(stats['metrics'])

    # Агрегируем результаты по всем файлам
    if all_times:
        try:
            std_dev = statistics.stdev(all_times)
            std_dev_algo = statistics.stdev(all_algorithm_times)
        except statistics.StatisticsError:
            std_dev = 0
            std_dev_algo = 0
        
        return {
            'times': all_times,
            'algorithm_times': all_algorithm_times,
            'avg_time': statistics.mean(all_times),
            'best_time': min(all_times),
            'std_dev': std_dev,
            # 'metrics': all_metrics
        }
    else:
        return {'times': [], 'algorithm_times': [], 'avg_time': float('inf'), 'best_time': float('inf'), 'std_dev': 0}

if __name__ == '__main__':
    main()