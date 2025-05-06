# main.py
import os
import sys
import time
import statistics
import glob  # Импортируем модуль glob для поиска файлов

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

def run_test(puzzle_file, num_runs=5, ModelClass=OriginalModel, SolverClass=OriginalSolver):
    """Запускает несколько прогонов решения головоломки и возвращает статистику."""
    times = []
    algorithm_times = []  # Время работы только алгоритма
    solutions = []
    statuses = []
    best_solution = None
    best_time = float('inf')
    best_algorithm_time = float('inf')

    for i in range(num_runs):
        try:
            puzzle = Puzzle.from_file(puzzle_file)
        except FileNotFoundError:
            print(f"File not found: {puzzle_file}")
            return [], [], [], [], None
        except Exception as e:
            print(f"Error loading puzzle from {puzzle_file}: {e}")
            return [], [], [], [], None

        start_time = time.time()
        model = ModelClass(puzzle)
        solver = SolverClass(model)
        algorithm_start_time = time.time()  # Засекаем время начала работы алгоритма

        # Вызываем правильный метод solve в зависимости от класса решателя
        if SolverClass == OriginalSolver:
            status, solution = solver.solve_with_cuts()
        elif SolverClass == FlowSolver:
            status, solution = solver.solve()
        else:
            print(f"Unknown SolverClass: {SolverClass}")
            return [], [], [], [], None

        algorithm_end_time = time.time()    # Засекаем время окончания работы алгоритма
        end_time = time.time()

        times.append(end_time - start_time)
        algorithm_times.append(algorithm_end_time - algorithm_start_time)
        solutions.append(solution)
        statuses.append(status)
        print(f"Run {i + 1}: Status = {status}, Total Time = {end_time - start_time:.4f} seconds, Algorithm Time = {algorithm_end_time - algorithm_start_time:.4f} seconds")

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

        if all_same:
            print("\nAll solutions are identical.")
        else:
            print("\nSolutions differ between runs!")
    else:
        print("\nNo optimal solutions found in any run.")

    if times:
        print(f"\n--- Statistics for {puzzle_file} ---")
        print(f"Number of runs: {num_runs}")
        print(f"Average time: {statistics.mean(times):.4f} seconds")
        print(f"Average algorithm time: {statistics.mean(algorithm_times):.4f} seconds")
        print(f"Median time: {statistics.median(times):.4f} seconds")
        print(f"Median algorithm time: {statistics.median(algorithm_times):.4f} seconds")
        print(f"Min time: {min(times):.4f} seconds")
        print(f"Min algorithm time: {min(algorithm_times):.4f} seconds")
        print(f"Max time: {max(times):.4f} seconds")
        print(f"Max algorithm time: {max(algorithm_times):.4f} seconds")
        if len(times) > 1:
            try:
                print(f"Standard deviation: {statistics.stdev(times):.4f} seconds")
                print(f"Standard deviation algorithm time: {statistics.stdev(algorithm_times):.4f} seconds")
            except statistics.StatisticsError:
                print("Standard deviation: Not applicable (only one data point)")

        return statuses, solutions, times, algorithm_times, best_solution


def compare_models(puzzle_files, num_runs=5, ModelClass1=OriginalModel, SolverClass1=OriginalSolver,
                   ModelClass2=FlowModel, SolverClass2=FlowSolver,
                   ModelClass3=ModifiedIterativeILP):
    """Сравнивает три модели на наборе головоломок."""
    results1 = {}
    results2 = {}
    results3 = {}
    
    print("Testing Model 1 (Original):")
    for puzzle_file in puzzle_files:
        print(f"\n--- Testing puzzle: {puzzle_file} ---")
        statuses, solutions, times, algorithm_times, best_solution = run_test(puzzle_file, num_runs, ModelClass1, SolverClass1)
        results1[puzzle_file] = {"statuses": statuses, "solutions": solutions, "times": times, "algorithm_times": algorithm_times,
                                  "best_solution": best_solution}
        if best_solution:
            print(f"Visualizing best solution for Model 1, puzzle {puzzle_file}")
            visualize_solution(puzzle_file, best_solution, title="Original Model")
        else:
            print(f"No solution found for Model 1, puzzle {puzzle_file}")

    print("\nTesting Model 2 (Flow):")
    for puzzle_file in puzzle_files:
        print(f"\n--- Testing puzzle: {puzzle_file} ---")
        statuses, solutions, times, algorithm_times, best_solution = run_test(puzzle_file, num_runs, ModelClass2, SolverClass2)
        results2[puzzle_file] = {"statuses": statuses, "solutions": solutions, "times": times, "algorithm_times": algorithm_times,
                                  "best_solution": best_solution}
        if best_solution:
            print(f"Visualizing best solution for Model 2, puzzle {puzzle_file}")
            visualize_solution(puzzle_file, best_solution, title="Flow Model")
        else:
            print(f"No solution found for Model 2, puzzle {puzzle_file}")

    print("\nTesting Model 3 (Modified Iterative):")
    for puzzle_file in puzzle_files:
        print(f"\n--- Testing puzzle: {puzzle_file} ---")
        times = []
        algorithm_times = []
        solutions = []
        statuses = []
        best_solution = None
        best_time = float('inf')
        best_algorithm_time = float('inf')

        for i in range(num_runs):
            try:
                puzzle = Puzzle.from_file(puzzle_file)
                solver = ModelClass3(puzzle)
                start_time = time.time()
                solution = solver.solve()
                end_time = time.time()
                
                times.append(end_time - start_time)
                algorithm_times.append(end_time - start_time)
                solutions.append(solution)
                
                if solution:
                    status = pywraplp.Solver.OPTIMAL
                    if is_solution_valid(puzzle, solution):
                        if end_time - start_time < best_algorithm_time:
                            best_time = end_time - start_time
                            best_algorithm_time = end_time - start_time
                            best_solution = solution
                else:
                    status = pywraplp.Solver.INFEASIBLE
                
                statuses.append(status)
                print(f"Run {i + 1}: Status = {status}, Total Time = {end_time - start_time:.4f} seconds, Algorithm Time = {end_time - start_time:.4f} seconds")
                
            except Exception as e:
                print(f"Error in run {i + 1}: {str(e)}")
                continue

        results3[puzzle_file] = {
            "statuses": statuses,
            "solutions": solutions,
            "times": times,
            "algorithm_times": algorithm_times,
            "best_solution": best_solution
        }
        
        if best_solution:
            print(f"Visualizing best solution for Model 3, puzzle {puzzle_file}")
            visualize_solution(puzzle_file, best_solution, title="Modified Iterative Model")
        else:
            print(f"No solution found for Model 3, puzzle {puzzle_file}")

        # Выводим статистику для Model 3
        if times:
            print(f"\n--- Statistics for Model 3, puzzle {puzzle_file} ---")
            print(f"Number of runs: {num_runs}")
            print(f"Average time: {statistics.mean(times):.4f} seconds")
            print(f"Average algorithm time: {statistics.mean(algorithm_times):.4f} seconds")
            print(f"Median time: {statistics.median(times):.4f} seconds")
            print(f"Median algorithm time: {statistics.median(algorithm_times):.4f} seconds")
            print(f"Min time: {min(times):.4f} seconds")
            print(f"Min algorithm time: {min(algorithm_times):.4f} seconds")
            print(f"Max time: {max(times):.4f} seconds")
            print(f"Max algorithm time: {max(algorithm_times):.4f} seconds")
            if len(times) > 1:
                try:
                    print(f"Standard deviation: {statistics.stdev(times):.4f} seconds")
                    print(f"Standard deviation algorithm time: {statistics.stdev(algorithm_times):.4f} seconds")
                except statistics.StatisticsError:
                    print("Standard deviation: Not applicable (only one data point)")

    print("\n--- Comparison Summary ---")
    for puzzle_file in puzzle_files:
        print(f"\nPuzzle: {puzzle_file}")
        
        print(f"Model 1 (Original):")
        if results1[puzzle_file]["times"]:
            avg_time1 = statistics.mean(results1[puzzle_file]['times'])
            avg_algorithm_time1 = statistics.mean(results1[puzzle_file]['algorithm_times'])
            print(f"Average total time: {avg_time1:.4f} seconds")
            print(f"Average algorithm time: {avg_algorithm_time1:.4f} seconds")
            num_optimal = sum(1 for status in results1[puzzle_file]["statuses"] if status == 0)
            print(f"Optimal solutions: {num_optimal}/{num_runs}")
        else:
            print("No solutions found.")

        print(f"Model 2 (Flow):")
        if results2[puzzle_file]["times"]:
            avg_time2 = statistics.mean(results2[puzzle_file]['times'])
            avg_algorithm_time2 = statistics.mean(results2[puzzle_file]['algorithm_times'])
            print(f"Average total time: {avg_time2:.4f} seconds")
            print(f"Average algorithm time: {avg_algorithm_time2:.4f} seconds")
            num_optimal = sum(1 for status in results2[puzzle_file]["statuses"] if status == 0)
            print(f"Optimal solutions: {num_optimal}/{num_runs}")
        else:
            print("No solutions found.")

        print(f"Model 3 (Modified Iterative):")
        if results3[puzzle_file]["times"]:
            avg_time3 = statistics.mean(results3[puzzle_file]['times'])
            avg_algorithm_time3 = statistics.mean(results3[puzzle_file]['algorithm_times'])
            print(f"Average total time: {avg_time3:.4f} seconds")
            print(f"Average algorithm time: {avg_algorithm_time3:.4f} seconds")
            num_optimal = sum(1 for status in results3[puzzle_file]["statuses"] if status == 0)
            print(f"Optimal solutions: {num_optimal}/{num_runs}")
        else:
            print("No solutions found.")


def visualize_solution(puzzle_file, solution, title="Hashiwokakero Solution"):
    """Визуализирует решение головоломки."""
    puzzle = Puzzle.from_file(puzzle_file) #  puzzle_file должен быть полным путем к файлу.
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
    ax.invert_yaxis()  # Инвертируем ось Y
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title)  # Используем передаваемый заголовок
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
            return False  # Если количество мостов не совпадает с требуемой степенью, решение неверное

    return True  # Если все острова имеют правильное количество мостов, решение верное

if __name__ == '__main__':
    # Находим все файлы .has в папке try
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'try')  # Путь к папке 'try'
    puzzle_files = glob.glob(os.path.join(folder_path, 'Hs_20_20_40_30_552.has'))  # Находим все файлы .has

    # Сравниваем три модели на всех найденных файлах
    compare_models(puzzle_files, num_runs=3)