import os
import sys
import time
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.puzzle import Puzzle
from src.core.model import Model
from src.core.solver import Solver
import pandas as pd
import matplotlib.pyplot as plt


def run_test(puzzle_file, num_runs=5):
    """Запускает несколько прогонов решения головоломки и возвращает статистику."""
    times = []
    solutions = []
    statuses = []

    for i in range(num_runs):
        start_time = time.time()
        puzzle = Puzzle.from_file(os.path.join(os.path.dirname(__file__), '..', 'examples', puzzle_file))
        model = Model(puzzle)
        solver = Solver(model)
        status, solution = solver.solve_with_cuts()
        end_time = time.time()
        print(f"Total program time = {end_time - start_time:.4f} seconds")

        times.append(end_time - start_time)
        solutions.append(solution)
        statuses.append(status)
        print(f"Run {i + 1}: Status = {status}, Time = {end_time - start_time:.4f} seconds")

    return statuses, solutions, times


def compare_models(puzzle_files, num_runs=5):
    """Сравнивает и визуализирует лучшее решение для каждой головоломки."""
    results = {}
    visualize = False

    for puzzle_file in puzzle_files:
        print(f"\n--- Testing puzzle: {puzzle_file} ---")
        statuses, solutions, times = run_test(puzzle_file, num_runs)

        # Находим самое быстрое решение для этой головоломки
        min_avg_time = float('inf')
        best_solution = None

        for i in range(len(solutions)):
            if solutions[i] is not None:
                avg_time = times[i]
                if avg_time < min_avg_time:
                    min_avg_time = avg_time
                    best_solution = solutions[i]

        if best_solution:
            print(f"\n--- The fastest solution for {puzzle_file} had time {min_avg_time:.4f} seconds ---")
            visualize_solution(puzzle_file, best_solution)
        else:
            print(f"\n--- No solution found for {puzzle_file} ---")


def visualize_solution(puzzle_file, solution):
    """Визуализирует решение головоломки."""
    puzzle = Puzzle.from_file(os.path.join(os.path.dirname(__file__), '..', 'examples', puzzle_file))
    islands = puzzle.get_all_islands()
    max_x = max(island['x'] for island in islands) + 1
    max_y = max(island['y'] for island in islands) + 1

    # Создаем DataFrame для удобства
    df = pd.DataFrame(islands)

    # Создаем два подграфика: один для начального состояния, другой для решения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max_x * 2, max_y))

    # Рисуем начальное состояние (только острова)
    for index, row in df.iterrows():
        ax1.plot(row['x'], row['y'], 'o', markersize=20, color='skyblue')
        ax1.text(row['x'], row['y'], str(row['degree']), ha='center', va='center', fontsize=12)
    
    ax1.set_xlim(-1, max_x)
    ax1.set_ylim(-1, max_y)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Initial Puzzle State')

    # Рисуем решение (острова и мосты)
    for index, row in df.iterrows():
        ax2.plot(row['x'], row['y'], 'o', markersize=20, color='skyblue')
        ax2.text(row['x'], row['y'], str(row['degree']), ha='center', va='center', fontsize=12)

    # Рисуем мосты
    for (i, j), bridges in solution.items():
        island1 = next(island for island in islands if island['id'] == i)
        island2 = next(island for island in islands if island['id'] == j)
        x1, y1 = island1['x'], island1['y']
        x2, y2 = island2['x'], island2['y']

        if bridges == 1:
            ax2.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=2)
        elif bridges == 2:
            ax2.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=4)

    ax2.set_xlim(-1, max_x)
    ax2.set_ylim(-1, max_y)
    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Solution')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Выбор режима: одиночная головоломка или сравнение
    mode = "compare"  # Или "single"

    if mode == "single":
        puzzle_file = '3.txt'
        print(f"--- Solving puzzle: {puzzle_file} ---")
        statuses, solutions, times = run_test(puzzle_file, num_runs=5)

        # Визуализация (если найдено решение)
        if solutions and solutions[0]:
            visualize_solution(puzzle_file, solutions[0])
        else:
            print("No solution found.")

    elif mode == "compare":
        puzzle_files = ['try\puzzle8.txt']  # Список файлов с головоломками для сравнения
        compare_models(puzzle_files, num_runs=5)

    else:
        print("Invalid mode. Choose 'single' or 'compare'.")