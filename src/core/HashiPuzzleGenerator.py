import random
import matplotlib.pyplot as plt
import pandas as pd

class HashiPuzzleGenerator:
    def __init__(self, n, d1, d2, alpha, beta):
        """
        Инициализирует генератор пазлов Hashi.

        Args:
            n (int): Желаемое количество островов.
            d1 (int): Ширина сетки.
            d2 (int): Высота сетки.
            alpha (float): Параметр, влияющий на количество циклов (0.0 - 1.0).
            beta (float): Параметр, влияющий на количество двойных мостов (0.0 - 1.0).
        """
        self.n = n
        self.d1 = d1  # width
        self.d2 = d2  # height
        self.alpha = alpha
        self.beta = beta
        self.grid = [[None for _ in range(d1)] for _ in range(d2)]  #  Изначально пустая сетка
        self.islands = []  #  Список островов (id, x, y)
        self.edges = {}  #  Словарь ребер (island1_id, island2_id): num_bridges
        self.next_island_id = 0
        self.puzzle = None # для хранения сгенерированного пазла

    def generate(self):
        """Генерирует пазл Hashi."""

        # Шаг 1: Размещение островов
        self.place_islands()

        # Шаг 2: Создание циклов
        self.create_cycles()

        # Шаг 3: Создание двойных мостов
        self.create_double_edges()

        # Шаг 4: Подсчет смежности и финальный пазл
        self.puzzle = self.finalize_puzzle()

        return self.puzzle

    def place_islands(self):
        """Размещает острова на сетке."""
        max_attempts = 1000000  # Уменьшаем количество попыток, но делаем алгоритм умнее
        attempts = 0
        
        # Размещаем первый остров в центре
        x = self.d1 // 2
        y = self.d2 // 2
        island_id = self.create_island(x, y)

        # Создаем сетку для отслеживания доступных позиций
        available_positions = set()
        for i in range(2, self.d1-2, 2):
            for j in range(2, self.d2-2, 2):
                available_positions.add((i, j))

        # Размещаем остальные острова
        while len(self.islands) < self.n and attempts < max_attempts:
            attempts += 1
            
            if not available_positions:
                    break
            
            # Выбираем случайную доступную позицию
            x, y = random.choice(list(available_positions))
            available_positions.remove((x, y))
                
                # Проверяем, можно ли добавить остров
                if self.is_valid_position(x, y):
                    new_island_id = self.create_island(x, y)
                    
                    # Пытаемся соединить с ближайшими существующими островами
                    connected = False
                    for existing_id in self.islands:
                        if existing_id != new_island_id:
                        ex_x, ex_y = self.get_island_coordinates(existing_id)
                        if self.can_connect(x, y, ex_x, ex_y):
                                self.edges[tuple(sorted((new_island_id, existing_id)))] = 1
                                connected = True
                                break
                    
                    # Если не удалось соединить, удаляем остров
                    if not connected:
                        self.grid[y][x] = None
                        self.islands.remove(new_island_id)
                        self.next_island_id -= 1
                    available_positions.add((x, y))

        if len(self.islands) < self.n:
            print(f"Предупреждение: Удалось создать только {len(self.islands)} островов из {self.n}")

    def create_island(self, x, y):
        """Создаёт остров в заданной позиции."""
        island_id = self.next_island_id
        self.next_island_id += 1
        self.islands.append(island_id)
        self.grid[y][x] = island_id  #  Пометить клетку как занятую островом
        return island_id

    def add_island_and_edge(self, existing_island_id, direction):
        """Добавляет новый остров и соединяет его с существующим."""
        existing_island_x, existing_island_y = self.get_island_coordinates(existing_island_id)
        new_x, new_y = self.find_new_island_position(existing_island_x, existing_island_y, direction)

        if new_x is not None and new_y is not None:
            # Проверяем, что вокруг нового острова достаточно свободного места
            if self.is_valid_position(new_x, new_y):
                new_island_id = self.create_island(new_x, new_y)
                # Создаем ребро между островами
                self.edges[tuple(sorted((existing_island_id, new_island_id)))] = 1
                return True
        return False

    def is_valid_position(self, x, y):
        """Проверяет, что позиция подходит для размещения острова."""
        # Проверяем, что клетка свободна
        if self.grid[y][x] is not None:
            return False
            
        # Проверяем только соседние клетки по диагонали
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.d1 and 0 <= ny < self.d2:
                if self.grid[ny][nx] is not None:
                    return False
        return True

    def get_island_coordinates(self, island_id):
        """Возвращает координаты острова по его ID."""
        for y in range(self.d2):
            for x in range(self.d1):
                if self.grid[y][x] == island_id:
                    return x, y
        return None, None  # Остров не найден

    def find_new_island_position(self, start_x, start_y, direction):
        """Находит позицию для нового острова в заданном направлении."""
        x, y = start_x, start_y

        if direction == 'top':
            y_range = range(start_y - 1, -1, -1)
            x_val = x
        elif direction == 'bottom':
            y_range = range(start_y + 1, self.d2)
            x_val = x
        elif direction == 'left':
            x_range = range(start_x - 1, -1, -1)
            y_val = y
        elif direction == 'right':
            x_range = range(start_x + 1, self.d1)
            y_val = y
        else:
            return None, None

        # Проверяем, не пересекаем ли существующие острова или ребра
        if direction in ['top', 'bottom']:
            if not y_range:
                return None, None

            # Пробуем найти ближайшую свободную позицию
            for new_y in y_range:
                if self.grid[new_y][x_val] is not None:
                    continue
                if self.is_valid_position(x_val, new_y):
                    return x_val, new_y

        elif direction in ['left', 'right']:
            if not x_range:
                return None, None

            # Пробуем найти ближайшую свободную позицию
            for new_x in x_range:
                if self.grid[y_val][new_x] is not None:
                    continue
                if self.is_valid_position(new_x, y_val):
                    return new_x, y_val

        return None, None

    def create_cycles(self):
        """Создает циклы, добавляя ребра."""
        num_cycles = int(self.alpha * self.n * 2)  # Увеличиваем количество циклов
        for _ in range(num_cycles):
            island1_id, island2_id = random.sample(self.islands, 2)
            if tuple(sorted((island1_id, island2_id))) not in self.edges:
                x1, y1 = self.get_island_coordinates(island1_id)
                x2, y2 = self.get_island_coordinates(island2_id)

                if self.can_connect(x1, y1, x2, y2):
                    self.edges[tuple(sorted((island1_id, island2_id)))] = 1

    def can_connect(self, x1, y1, x2, y2):
        """Проверяет, можно ли соединить два острова."""
        if x1 != x2 and y1 != y2:
            return False  # Не горизонтально и не вертикально

        if x1 == x2:  # Соединяем вертикально
            for y in range(min(y1, y2) + 1, max(y1, y2)):
                if self.grid[y][x1] is not None:
                    return False  # Пересечение с другим островом
        else:  # Соединяем горизонтально
            for x in range(min(x1, x2) + 1, max(x1, x2)):
                if self.grid[y1][x] is not None:
                    return False  # Пересечение с другим островом
        return True  # Можно соединить

    def create_double_edges(self):
        """Преобразует одинарные ребра в двойные с вероятностью beta."""
        for edge, bridges in self.edges.items():
            if random.random() < self.beta:
                self.edges[edge] = 2  # Превращаем одинарный мост в двойной

    def finalize_puzzle(self):
        """Подсчитывает смежность и возвращает финальный пазл."""

        # Создаем словарь для хранения степени каждого острова
        island_degrees = {island_id: 0 for island_id in self.islands}

        # Подсчитываем количество мостов для каждого острова
        for (island1_id, island2_id), num_bridges in self.edges.items():
            island_degrees[island1_id] += num_bridges
            island_degrees[island2_id] += num_bridges

        #  Создаем структуру данных для пазла (адаптируйте под свой формат)
        puzzle = {
            'width': self.d1,
            'height': self.d2,
            'islands': [{'id': island_id, 'x': self.get_island_coordinates(island_id)[0], 'y': self.get_island_coordinates(island_id)[1], 'degree': island_degrees[island_id]}
                        for island_id in self.islands],
            'edges': self.edges  #  Храним информацию о ребрах для отладки
        }

        return puzzle

    def visualize_solution(self, solution, title="Hashiwokakero Solution"):
        """Визуализирует решение головоломки."""
        if self.puzzle is None:
            print("Error: No puzzle generated yet. Call generate() first.")
            return

        islands = self.puzzle['islands']
        if not islands:
            print("Error: No islands found in the puzzle.")
            return

        # Определяем размеры сетки
        max_x = self.d1 #  Используем сохраненные размеры
        max_y = self.d2

        # Создаем DataFrame для удобства
        df = pd.DataFrame(islands)

        # Создаем Figure и Axes
        fig, ax = plt.subplots(figsize=(max_x, max_y))

        # Рисуем острова
        for index, row in df.iterrows():
            ax.plot(row['x'], row['y'], 'o', markersize=20, color='skyblue')
            ax.text(row['x'], row['y'], str(row['degree']), ha='center', va='center', fontsize=12)

        # Рисуем мосты
        if solution: # Проверяем, что решение не пустое
            for (i, j), bridges in solution.items():
                # Находим острова по ID
                try:
                    island1 = next(island for island in islands if island['id'] == i)
                    island2 = next(island for island in islands if island['id'] == j)
                    x1, y1 = island1['x'], island1['y']
                    x2, y2 = island2['x'], island2['y']

                    if bridges == 1:
                        ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=2)
                    elif bridges == 2:
                        ax.plot([x1, x2], [y1, y2], '-', color='gray', linewidth=4)
                except StopIteration:
                    print(f"Warning: Island with ID {i} or {j} not found in puzzle.") # Обработка ситуации, если ID в решении нет в островах
                    continue

        # Настраиваем график
        ax.set_xlim(-1, max_x)
        ax.set_ylim(-1, max_y)
        ax.invert_yaxis()  # Инвертируем ось Y
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(title)  # Используем передаваемый заголовок
        plt.show()

#  Пример использования (адаптируйте параметры)
generator = HashiPuzzleGenerator(n=50, d1=16, d2=16, alpha=0.5, beta=0.5)
puzzle = generator.generate()


#  Пример использования визуализации (после решения головоломки):
#  Предполагается, что у тебя есть переменная solution с решением
solution = {} #  Замени на реальное решение
generator.visualize_solution(solution, title="Generated Hashi Puzzle")