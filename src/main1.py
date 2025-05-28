# main.py
import os
import sys
import time
import statistics
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    """Генерирует расширенную LaTeX таблицу с результатами тестирования."""
    latex_table = """\\begin{table}[h]
\\centering
\\caption{Результаты тестирования алгоритмов на различных головоломках}
\\begin{tabular}{|l|c|c|c|c|c|c|c|}
\\hline
\\textbf{Головоломка} & \\textbf{Модель} & \\textbf{Время (с)} & \\textbf{Мосты} & \\textbf{Связность} & \\textbf{Плотность} & \\textbf{Макс. мостов} & \\textbf{Ст. откл. (с)} \\\\
\\hline
"""
    
    for puzzle_name, models in results.items():
        for i, (model_name, stats) in enumerate(models.items()):
            metrics = stats.get('metrics', [{}])
            avg_metrics = {
                'bridge_count': statistics.mean([m.get('bridge_count', 0) for m in metrics]),
                'island_connectivity': statistics.mean([m.get('island_connectivity', 0) for m in metrics]),
                'solution_density': statistics.mean([m.get('solution_density', 0) for m in metrics]),
                'max_bridges_per_island': statistics.mean([m.get('max_bridges_per_island', 0) for m in metrics])
            }
            
            if i == 0:
                latex_table += f"\\multirow{{3}}{{*}}{{{puzzle_name}}} & {model_name} & {stats['avg_time']:.4f} & {avg_metrics['bridge_count']:.1f} & {avg_metrics['island_connectivity']:.2f} & {avg_metrics['solution_density']:.2f} & {avg_metrics['max_bridges_per_island']:.1f} & {stats['std_dev']:.4f} \\\\\n"
        else:
                latex_table += f" & {model_name} & {stats['avg_time']:.4f} & {avg_metrics['bridge_count']:.1f} & {avg_metrics['island_connectivity']:.2f} & {avg_metrics['solution_density']:.2f} & {avg_metrics['max_bridges_per_island']:.1f} & {stats['std_dev']:.4f} \\\\\n"
        latex_table += "\\hline\n"
    
    latex_table += """\\end{tabular}
\\end{table}"""
    
    return latex_table

def visualize_solution(puzzle_file, solution, title="Hashiwokakero Solution"):
    """Визуализирует решение головоломки."""
    puzzle = Puzzle.from_file(puzzle_file)
    islands = puzzle.get_all_islands()
    max_x = max(island['x'] for island in islands) + 1
    max_y = max(island['y'] for island in islands) + 1

    # Создаем DataFrame для удобства
    df = pd.DataFrame(islands)

    # Создаем Figure и Axes
    fig, ax = plt.subplots(figsize=(max_x, max_y))

    # Рисуем острова
    for index, row in df.iterrows():
        ax.plot(row['x'], row['y'], 'o', markersize=20, color='skyblue')
        ax.text(row['x'], row['y'], str(row['degree']), ha='center', va='center', fontsize=12)

    # Рисуем мосты
    for (i, j), bridges in solution.items():
        island1 = next(island for island in islands if island['id'] == i)
        island2 = next(island for island in islands if island['id'] == j)
        x1, y1 = island1['x'], island1['y']
        x2, y2 = island2['x'], island2['y']

        if bridges == 1:
            ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=2)
        elif bridges == 2:
            ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=4)

    # Настраиваем график
    ax.set_xlim(-1, max_x)
    ax.set_ylim(-1, max_y)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title)
    plt.show()

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

if __name__ == '__main__':
    # Выбор папки с головоломками
    folder_path = select_puzzle_folder()
    folder_name = os.path.basename(folder_path)
    
    # Ищем все .has и .txt файлы, включая подпапки
    puzzle_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.has', '.txt')):
                puzzle_files.append(os.path.join(root, file))
    
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