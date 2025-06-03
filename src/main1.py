# main.py
import os
import sys
import time
import statistics
import glob
import argparse

# Add both src and hashiwokakero to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hashiwokakero')))

from src.core.puzzle import Puzzle

# Импортируем FlowModel и FlowSolver
from src.core.model_flow import FlowModel
from src.core.solver_flow import FlowSolver

import networkx as nx
from ortools.linear_solver import pywraplp

# Импортируем оригинальные Model и Solver
from src.core.model import Model as OriginalModel
from src.core.solver import Solver as OriginalSolver

# Импортируем ModifiedIterativeILP
from src.core.ModifiedIterativeILP import ModifiedIterativeILP

# Импортируем функции для генерации головоломок
from hashi.generator import generate_till_full, generate_solvable_puzzle
from hashi.export import save_grid

import pandas as pd
import matplotlib.pyplot as plt

def select_puzzle_folder():
    """Позволяет пользователю выбрать папку с головоломками."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'examples')
    # Добавим опцию для выбора папки output из hashi-generator
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hashi-generator', 'output'))
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    folders.append('hashi-generator/output')  # Добавляем опцию для output
    
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
    
    # Количество мостов
    bridge_count = sum(solution.values())
    
    # Подсчет связности островов
    connected_islands = set()
    for (i, j), bridges in solution.items():
        if bridges > 0:
            connected_islands.add(i)
            connected_islands.add(j)
    island_connectivity = len(connected_islands) / total_islands
    
    # Плотность решения
    solution_density = bridge_count / total_possible_bridges if total_possible_bridges > 0 else 0
    
    # Максимальное количество мостов на один остров
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

def run_test(puzzle_file, num_runs=5, ModelClass=OriginalModel, SolverClass=OriginalSolver):
    """Запускает несколько прогонов решения головоломки и возвращает расширенную статистику."""
    times = []
    algorithm_times = []
    solutions = []
    statuses = []
    best_solution = None
    best_time = float('inf')
    best_algorithm_time = float('inf')
    metrics = []

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
        
        # Специальная обработка для ModifiedIterativeILP
        if ModelClass == ModifiedIterativeILP:
            solver = ModelClass(puzzle)
            algorithm_start_time = time.time()
            solution = solver.solve()
            algorithm_end_time = time.time()
            end_time = time.time()
            
            times.append(end_time - start_time)
            algorithm_times.append(algorithm_end_time - algorithm_start_time)
            solutions.append(solution)
            status = pywraplp.Solver.OPTIMAL if solution else pywraplp.Solver.INFEASIBLE
            statuses.append(status)
            
            if solution:
                metrics.append(calculate_metrics(puzzle, solution))
            
            if solution and is_solution_valid(puzzle, solution):
                if algorithm_end_time - algorithm_start_time < best_algorithm_time:
                    best_time = end_time - start_time
                    best_algorithm_time = algorithm_end_time - algorithm_start_time
                    best_solution = solution
                    # Визуализируем решение
                    visualize_solution(puzzle_file, best_solution, title=f"Solution for {os.path.basename(puzzle_file)}")
        else:
            model = ModelClass(puzzle)
            solver = SolverClass(model)
            algorithm_start_time = time.time()

            # Вызываем правильный метод solve в зависимости от класса решателя
            if SolverClass == OriginalSolver:
                status, solution = solver.solve_with_cuts()
            elif SolverClass == FlowSolver:
                status, solution = solver.solve()
            else:
                print(f"Unknown SolverClass: {SolverClass}")
                return {'statuses': [], 'solutions': [], 'times': [], 'algorithm_times': [], 'best_solution': None, 'metrics': []}

            algorithm_end_time = time.time()
            end_time = time.time()

            times.append(end_time - start_time)
            algorithm_times.append(algorithm_end_time - algorithm_start_time)
            solutions.append(solution)
            statuses.append(status)
            
            if solution:
                metrics.append(calculate_metrics(puzzle, solution))

            if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                if solution and is_solution_valid(puzzle, solution):
                    if algorithm_end_time - algorithm_start_time < best_algorithm_time:
                        best_time = end_time - start_time
                        best_algorithm_time = algorithm_end_time - algorithm_start_time
                        best_solution = solution
                        # Визуализируем решение
                        visualize_solution(puzzle_file, best_solution, title=f"Solution for {os.path.basename(puzzle_file)}")
                else:
                    print("Invalid Solution")

    optimal_solutions = [s for s in solutions if s is not None]
    if optimal_solutions:
        first_solution = optimal_solutions[0]
        all_same = all(s == first_solution for s in optimal_solutions)

    if times:
        if len(times) > 1:
            try:
                std_dev = statistics.stdev(times)
                std_dev_algo = statistics.stdev(algorithm_times)
            except statistics.StatisticsError:
                std_dev = 0
                std_dev_algo = 0

    return {
        'statuses': statuses,
        'solutions': solutions,
        'times': times,
        'algorithm_times': algorithm_times,
        'best_solution': best_solution,
        'metrics': metrics
    }

def generate_latex_table(results):
    """Генерирует LaTeX таблицу с результатами."""
    if not results:
        return "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\nNo results available\n\\hline\n\\end{tabular}"

    # Группируем результаты по размеру сетки
    grid_sizes = {}
    for puzzle_name, puzzle_results in results.items():
        # Парсим размер из имени файла (формат: Hs_16_16_...)
        size_parts = puzzle_name.split('_')[1:3]  # Получаем ['16', '16']
        size = int(size_parts[0])  # Берем первое число как размер
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
    table += "Grid Size & Best Time & Original & Flow & Modified \\\\\n\\hline\n"
    table += "\n".join(rows)
    table += "\n\\hline\n\\end{tabular}"
    
    return table

def visualize_solution(puzzle_file, solution, title="Hashiwokakero Solution"):
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
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'generator_results')
    os.makedirs(results_dir, exist_ok=True) # Убедимся, что директория существует
    
    puzzle_name = os.path.basename(puzzle_file).replace('.has', '').replace('.txt', '')
    image_path = os.path.join(results_dir, f'{puzzle_name}_solution.png')
    plt.savefig(image_path)
    print(f"Решение сохранено как изображение: {image_path}")
    plt.close(fig)

def is_solution_valid(puzzle, solution):
    """Проверяет, что решение удовлетворяет требованиям по количеству мостов для каждого острова."""
    islands = puzzle.get_all_islands()
    for island in islands:
        island_id = island['id']
        degree = island['degree']

        # Считаем количество мостов, прилегающих к острову
        bridges_count = 0
        for (i, j), bridges in solution.items():
            if i == island_id or j == island_id:
                bridges_count += bridges

        if bridges_count != degree:
            return False

    return True

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
    """Показывает главное меню программы."""
    print("\nГлавное меню:")
    print("1. Создать новые головоломки")
    print("2. Протестировать существующие головоломки")
    print("3. Выйти")
    
    while True:
        try:
            choice = int(input("\nВыберите действие (1-3): "))
            if 1 <= choice <= 3:
                return choice
            print("Неверный выбор. Попробуйте снова.")
        except ValueError:
            print("Пожалуйста, введите число.")

def generate_puzzles():
    """Генерирует новые головоломки используя генератор из hashiwokakero."""
    # Параметры для генерации
    test_cases = [
        # {'w': 16, 'h': 16, 'n': 40},
        # {'w': 16, 'h': 16, 'n': 60},
        # {'w': 16, 'h': 16, 'n': 80},
        {'w': 16, 'h': 16, 'n': 100}
    ]
    
    # Путь к директории для результатов
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'generator_results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nЗапуск генератора...")
    try:
        for case in test_cases:
            print(f"\nГенерация пазла {case['w']}x{case['h']} с {case['n']} островами...")
            
            # Генерируем пазл используя генератор из hashiwokakero
            grid = generate_solvable_puzzle(case['w'], case['h'], case['n'], target_double_bridges=100)
            
            # Создаем имя файла
            puzzle_name = f"Hs_{case['w']}_{case['n']}_25_00"
            puzzle_dir = os.path.join(results_dir, puzzle_name)
            os.makedirs(puzzle_dir, exist_ok=True)
            
            # Находим следующий доступный индекс
            existing_files = [f for f in os.listdir(puzzle_dir) if f.endswith('.has')]
            next_index = len(existing_files) + 1
            
            # Сохраняем пазл
            puzzle_path = os.path.join(puzzle_dir, f"{puzzle_name}_{str(next_index).zfill(3)}.has")
            save_grid(grid, puzzle_path)
            
            print(f"Пазл сохранен в {puzzle_path}")
            
        print("\nВсе пазлы сгенерированы успешно!")
        return True
    except Exception as e:
        print(f"\nОшибка при генерации пазлов: {e}")
        return False

def main():
    # Добавляем парсер аргументов
    parser = argparse.ArgumentParser(description='Hashiwokakero puzzle generator and tester')
    parser.add_argument('--auto', action='store_true', help='Run in automatic mode')
    parser.add_argument('--test-folder', type=str, help='Folder to test in automatic mode')
    parser.add_argument('--puzzle', type=str, help='Path to a single puzzle file to test')
    args = parser.parse_args()
    
    # Если запущен в автоматическом режиме
    if args.auto:
        if args.test_folder:
            # Тестируем указанную папку
            folder_path = args.test_folder
            folder_name = os.path.basename(folder_path)
            
            # Ищем все .has и .txt файлы
            puzzle_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.has', '.txt')):
                        puzzle_files.append(os.path.join(root, file))
            
            if not puzzle_files:
                print(f"\nВ папке {folder_name} не найдено головоломок для тестирования.")
                return
            
            results = {}
            
            print(f"\nРезультаты тестирования алгоритмов для папки: {folder_name}")
            print("=" * 80)
            
            for puzzle_file in puzzle_files:
                puzzle_name = os.path.basename(puzzle_file)
                results[puzzle_name] = {}
                
                print(f"\nТестирование головоломки: {puzzle_name}")
                print("-" * 50)
                
                # Тестируем все три модели
                models = [
                    ('Original', OriginalModel, OriginalSolver),
                    ('Flow', FlowModel, FlowSolver),
                    ('Modified', ModifiedIterativeILP, None)
                ]
                
                for model_name, ModelClass, SolverClass in models:
                    stats = run_test(puzzle_file, num_runs=2, ModelClass=ModelClass, SolverClass=SolverClass)
                    
                    if stats['times']:
                        results[puzzle_name][model_name] = {
                            'avg_time': statistics.mean(stats['times']),
                            'avg_algo_time': statistics.mean(stats['algorithm_times']),
                            'std_dev': statistics.stdev(stats['times']) if len(stats['times']) > 1 else 0,
                            'metrics': stats['metrics']
                        }
            
            # Выводим общую сводку результатов и сохраняем в файл
            summary_file = print_summary(results, folder_name)
            
            # Генерируем и сохраняем LaTeX таблицу
            latex_table = generate_latex_table(results)
            latex_file = f'results_{folder_name}.tex'
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            
            print(f"\nТестирование завершено.")
            print(f"Результаты сохранены в файлы:")
            print(f"- {latex_file} (LaTeX таблица)")
            print(f"- {summary_file} (Общая сводка)")
        return
    
    # Если указан конкретный файл головоломки
    if args.puzzle:
        puzzle_file = args.puzzle
        if not os.path.exists(puzzle_file):
            print(f"Файл головоломки не найден: {puzzle_file}")
            return
            
        print(f"\nТестирование головоломки: {os.path.basename(puzzle_file)}")
        print("-" * 50)
        
        # Используем только Flow модель для проверки решаемости
        stats = run_test(puzzle_file, num_runs=1, ModelClass=FlowModel, SolverClass=FlowSolver)
        
        if stats['times']:
            print(f"\nРезультаты для модели Flow:")
            print(f"Время решения: {statistics.mean(stats['times']):.4f} сек")
            if stats['best_solution']:
                print("Решение найдено")
            else:
                print("Решение не найдено")
        return
    
    # Обычный интерактивный режим
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Создание новых головоломок
            if generate_puzzles():
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
            
            for puzzle_file in puzzle_files:
                puzzle_name = os.path.basename(puzzle_file)
                results[puzzle_name] = {}
                
                print(f"\nТестирование головоломки: {puzzle_name}")
                print("-" * 50)
                
                # Тестируем все три модели
                models = [
                    ('Original', OriginalModel, OriginalSolver),
                    ('Flow', FlowModel, FlowSolver),
                    ('Modified', ModifiedIterativeILP, None)
                ]
                
                for model_name, ModelClass, SolverClass in models:
                    stats = run_test(puzzle_file, num_runs=2, ModelClass=ModelClass, SolverClass=SolverClass)
                    
                    if stats['times']:
                        results[puzzle_name][model_name] = {
                            'avg_time': statistics.mean(stats['times']),
                            'avg_algo_time': statistics.mean(stats['algorithm_times']),
                            'std_dev': statistics.stdev(stats['times']) if len(stats['times']) > 1 else 0,
                            'metrics': stats['metrics']
                        }
            
            # Выводим общую сводку результатов и сохраняем в файл
            summary_file = print_summary(results, folder_name)
            
            # Генерируем и сохраняем LaTeX таблицу
            latex_table = generate_latex_table(results)
            latex_file = f'results_{folder_name}.tex'
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            
            print(f"\nТестирование завершено.")
            print(f"Результаты сохранены в файлы:")
            print(f"- {latex_file} (LaTeX таблица)")
            print(f"- {summary_file} (Общая сводка)")
        
        elif choice == 3:
            print("\nДо свидания!")
            break

if __name__ == '__main__':
    main()