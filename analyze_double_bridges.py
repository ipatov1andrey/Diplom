import os
from src.core.puzzle import Puzzle
from src.core.model import Model as OriginalModel
from src.core.solver import Solver as OriginalSolver

def count_bridge_links(grid):
    """Возвращает (total_links, double_links) по связям. Работает с нормализованной сеткой (Cell/Node objects)."""
    w, h = len(grid), len(grid[0])
    links = set()
    double_links = set()
    
    # Собираем координаты всех островов из нормализованной сетки
    islands_coords = []
    for i in range(w):
        for j in range(h):
            # Остров имеет n_type == 1
            if grid[i][j].n_type == 1:
                islands_coords.append((i, j))

    # Проверяем связи между всеми парами островов
    for i1_coords in islands_coords:
        for i2_coords in islands_coords:
            # Убедимся, что i1_coords < i2_coords для уникальных пар
            if i1_coords >= i2_coords: continue

            r1, c1 = i1_coords
            r2, c2 = i2_coords

            # Проверяем, лежат ли на одной горизонтали или вертикали
            if r1 == r2 or c1 == c2:
                # Определяем координаты ячеек между островами
                bridge_path_coords = []
                if r1 == r2: # Горизонталь
                    for c in range(min(c1, c2) + 1, max(c1, c2)):
                        bridge_path_coords.append((r1, c))
                else: # Вертикаль
                    for r in range(min(r1, r2) + 1, max(r1, r2)):
                        bridge_path_coords.append((r, c1))

                # Проверяем, все ли ячейки на пути являются мостами одного типа и толщины в нормализованной сетке
                if not bridge_path_coords: continue # Острова соседние, что не должно быть в валидной задаче Hashi

                is_valid_bridge_path = True
                bridge_thickness = 0
                
                # Проверяем, что все ячейки на пути - мосты (n_type == 2)
                if not all(grid[r][c].n_type == 2 for r, c in bridge_path_coords):
                    is_valid_bridge_path = False
                else:
                    # У всех мостов на пути должна быть одинаковая толщина (b_thickness)
                    thicknesses = set(grid[r][c].b_thickness for r, c in bridge_path_coords)
                    if len(thicknesses) != 1:
                        is_valid_bridge_path = False # Мосты разной толщины на одном пути - невалидно
                    else:
                        bridge_thickness = thicknesses.pop()

                if is_valid_bridge_path:
                    # Уникальный ключ для связи (пара отсортированных координат островов)
                    link_key = tuple(sorted([i1_coords, i2_coords]))
                    links.add(link_key)
                    if bridge_thickness == 2:
                        double_links.add(link_key)

    return len(links), len(double_links)

def normalize_grid(grid):
    """Преобразует входную сетку (list[list[int]] or list[list[Node]]) в сетку Cell-подобных объектов."""
    # Если уже Node-подобные объекты — преобразуем в Cell, чтобы гарантировать наличие нужных атрибутов
    if hasattr(grid[0][0], 'n_type'):
        class CellFromNode:
            def __init__(self, node):
                self.n_type = node.n_type
                # Для мостов, извлекаем b_thickness
                if node.n_type == 2:
                    self.b_thickness = node.b_thickness
                else:
                    self.b_thickness = 0
        return [[CellFromNode(val) for val in row] for row in grid]
    # Иначе — преобразуем из чисел
    class CellFromInt:
        def __init__(self, val):
            if isinstance(val, (int, float)):
                if val > 0:
                    self.n_type = 1  # остров
                    self.b_thickness = 0
                elif val == 0:
                    self.n_type = 0  # пусто
                    self.b_thickness = 0
                else:
                    self.n_type = 2  # мост
                    # Для мостов, отрицательное значение кодирует тип (-1 гориз. сингл, -2 гориз. дабл, -3 верт. сингл, -4 верт. дабл)
                    self.b_thickness = 2 if abs(val) in (2, 4) else 1
            else: # Неизвестный тип
                self.n_type = -1 # Обозначим как неизвестный тип
                self.b_thickness = 0
                # print(f"Warning: normalize_grid encountered unexpected value type: {type(val)} with value {val}") # Закомментируем, чтобы не засорять вывод

    return [[CellFromInt(val) for val in row] for row in grid]

def build_grid_from_solution_dict(puzzle, solution):
    """Строит числовую сетку-представление из словаря решения."""
    w, h = puzzle.width, puzzle.height
    # Создаем пустую сетку (заполняем нулями)
    grid_repr = [[0 for _ in range(h)] for _ in range(w)]

    # Размещаем острова (берем их из исходной головоломки), используя их степени
    for island in puzzle.get_all_islands():
        grid_repr[island['y']][island['x']] = island['degree']

    # Размещаем мосты на основе словаря решения
    # Словарь решения: {(island_id1, island_id2): bridges_count}
    for (id1, id2), bridges in solution.items():
        # Находим координаты островов по их ID
        island1 = None
        island2 = None
        for item in puzzle.get_all_islands():
            if item['id'] == id1:
                island1 = item
            if item['id'] == id2:
                island2 = item
            if island1 and island2: break

        if island1 and island2:
            x1, y1 = island1['x'], island1['y']
            x2, y2 = island2['x'], island2['y']

            # Определяем направление и значение моста для числовой сетки (-1,-2,-3,-4)
            bridge_val = 0
            if x1 == x2: # Вертикальный мост
                bridge_val = -3 if bridges == 1 else -4
                # Заполняем клетки между островами
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    grid_repr[y][x1] = bridge_val
            elif y1 == y2: # Горизонтальный мост
                bridge_val = -1 if bridges == 1 else -2
                # Заполняем клетки между островами
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    grid_repr[y1][x] = bridge_val

    return grid_repr # Возвращаем числовую сетку

def analyze_folder(folder):
    print(f"Анализ папки: {folder}")
    print("=" * 40)
    for fname in os.listdir(folder):
        if fname.endswith('.has'):
            path = os.path.join(folder, fname)
            print(f"Обработка файла: {fname}")
            try:
                puzzle = Puzzle.from_file(path)
                
                # --- Отладочный вывод для проверки puzzle.grid ---\n                print(f"  Тип puzzle.grid: {type(puzzle.grid)}")\n                if isinstance(puzzle.grid, list) and puzzle.grid:\n                     print(f"  Тип первого элемента puzzle.grid[0][0]: {type(puzzle.grid[0][0])}")\n                     # Выведем небольшой фрагмент сетки для визуальной проверки\n                     for row in puzzle.grid[:min(5, len(puzzle.grid))] : # Вывести первые 5 строк\n                         print("  ", row[:min(10, len(row))]) # Вывести первые 10 элементов строки\n                # --- Конец отладочного вывода ---\n

                # Анализ исходной задачи: нормализуем и считаем мосты
                grid = normalize_grid(puzzle.grid)
                total_links, double_links = count_bridge_links(grid)
                percent_gen = (double_links / total_links * 100) if total_links > 0 else 0

                # Решаем задачу оригинальной моделью
                model = OriginalModel(puzzle)
                solver = OriginalSolver(model)
                status, solution, _ = solver.solve_with_cuts()
                
                percent_sol = None
                if solution is not None:
                    # Строим числовую сетку из словаря решения, нормализуем ее и считаем мосты
                    solved_grid_repr = build_grid_from_solution_dict(puzzle, solution)
                    solved_grid = normalize_grid(solved_grid_repr)
                    total_links_s, double_links_s = count_bridge_links(solved_grid)
                    percent_sol = (double_links_s / total_links_s * 100) if total_links_s > 0 else 0

                print(f"  В задаче: {percent_gen:.2f}% двойных мостов по связям")
                if percent_sol is not None:
                    print(f"  В решении: {percent_sol:.2f}% двойных мостов по связям")
                else:
                    print("  Решение не найдено")
            except Exception as e:
                print(f"  Ошибка при обработке файла {fname}: {e}")
            print("-" * 40)

if __name__ == "__main__":
    # Укажите путь к папке с задачами
    folder = "examples/generated/16x16_80_75"
    analyze_folder(folder) 