from src.core.puzzle import Puzzle
import networkx as nx

def is_solution_valid(puzzle, solution):
    """Проверяет, что решение удовлетворяет всем требованиям:
    1. Количество мостов соответствует степени каждого острова
    2. Все острова соединены (граф связный)
    3. Мосты не пересекаются и не проходят через другие острова
    """
    if solution is None:
        print("DEBUG: Solution is None")
        return False
        
    islands = puzzle.get_all_islands()
    island_positions = {island['id']: (island['x'], island['y']) for island in islands}
    
    # 1. Проверяем количество мостов для каждого острова
    for island in islands:
        island_id = island['id']
        degree = island['degree']

        bridges_count = 0
        for (i, j), bridges in solution.items():
            if i == island_id or j == island_id:
                bridges_count += bridges

        if bridges_count != degree:
            print(f"DEBUG: Island {island_id} has incorrect bridge count: expected {degree}, got {bridges_count}")
            return False

    # 2. Проверяем связность графа
    G = nx.Graph()
    for island in islands:
        G.add_node(island['id'])
    for (i, j), bridges in solution.items():
        if bridges > 0:
            G.add_edge(i, j)
    if not nx.is_connected(G):
        print("DEBUG: Graph is not connected")
        return False

    # 3. Проверяем корректность мостов
    for (i, j), bridges in solution.items():
        if bridges > 0:
            pos1 = island_positions[i]
            pos2 = island_positions[j]
            
            # Проверяем, что острова на одной линии
            if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
                print(f"DEBUG: Islands {i} and {j} are not in line: {pos1} and {pos2}")
                return False
                
            # Проверяем, нет ли других островов между ними
            if pos1[0] == pos2[0]:  # Вертикальный мост
                start_y = min(pos1[1], pos2[1]) + 1
                end_y = max(pos1[1], pos2[1])
                for y in range(start_y, end_y):
                    if puzzle.grid[y][pos1[0]] != 0:
                        print(f"DEBUG: Found island between vertical bridge {i}-{j} at position ({pos1[0]}, {y})")
                        return False
            else:  # Горизонтальный мост
                start_x = min(pos1[0], pos2[0]) + 1
                end_x = max(pos1[0], pos2[0])
                for x in range(start_x, end_x):
                    if puzzle.grid[pos1[1]][x] != 0:
                        print(f"DEBUG: Found island between horizontal bridge {i}-{j} at position ({x}, {pos1[1]})")
                        return False

    return True 