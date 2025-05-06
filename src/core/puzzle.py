class Puzzle:
    def __init__(self, width, height, grid):
        self.width = width
        self.height = height
        self.grid = grid
        self.islands = self._find_islands()
        self.adjacency = self._build_adjacency()

    def _find_islands(self):
        islands = []
        island_id = 0
        for y in range(self.height):
            for x in range(self.width):
                degree = self.grid[y][x]
                if degree > 0:
                    islands.append({'id': island_id, 'x': x, 'y': y, 'degree': degree})
                    island_id += 1
        return islands

    def _build_adjacency(self):
        adj = {island['id']: [] for island in self.islands}
        for i in range(len(self.islands)):
            for j in range(i + 1, len(self.islands)):
                island1 = self.islands[i]
                island2 = self.islands[j]
                if self.is_neighbor(island1, island2):
                    adj[island1['id']].append(island2['id'])
                    adj[island2['id']].append(island1['id'])
        return adj

    def get_all_islands(self):
        return self.islands

    def get_adjacency(self):
        return self.adjacency

    def is_neighbor(self, island1, island2):
        if island1['x'] == island2['x']:
            # Check vertical neighbor
            blocked = False
            start_y = min(island1['y'], island2['y']) + 1
            end_y = max(island1['y'], island2['y'])
            for y in range(start_y, end_y):
                if self.grid[y][island1['x']] != 0:  # Проверяем, есть ли остров между ними
                    blocked = True
                    break
            return not blocked
        elif island1['y'] == island2['y']:
            # Check horizontal neighbor
            blocked = False
            start_x = min(island1['x'], island2['x']) + 1
            end_x = max(island1['x'], island2['x'])
            for x in range(start_x, end_x):
                if self.grid[island1['y']][x] != 0:  # Проверяем, есть ли остров между ними
                    blocked = True
                    break
            return not blocked
        return False

    def get_all_edges(self):
        edges = set()
        for island1 in self.islands:
            for island2 in self.islands:
                if island1['id'] != island2['id'] and self.is_neighbor(island1, island2):
                    id1, id2 = min(island1['id'], island2['id']), max(island1['id'], island2['id'])
                    edges.add((id1, id2))
        return list(edges)

    def get_intersections(self):
      intersections = set()
      edges = self.get_all_edges()

      for idx, (i, j) in enumerate(edges):
        island_i = next(island for island in self.islands if island['id'] == i)
        island_j = next(island for island in self.islands if island['id'] == j)

        for kdx in range(idx + 1, len(edges)):
          k, l = edges[kdx]
          island_k = next(island for island in self.islands if island['id'] == k)
          island_l = next(island for island in self.islands if island['id'] == l)

          # Проверка, являются ли ребра разными и не имеют общих вершин
          if {i, j}.intersection({k, l}) or (i == k or i == l or j == k or j == l):
            continue

          # Проверка горизонтальных и вертикальных пересечений
          if (island_i['x'] == island_j['x'] and island_k['y'] == island_l['y']):
            # Вертикальное ребро (i, j) и горизонтальное ребро (k, l)
            if (min(island_i['y'], island_j['y']) < island_k['y'] < max(island_i['y'], island_j['y']) and
                min(island_k['x'], island_l['x']) < island_i['x'] < max(island_k['x'], island_l['x'])):
              intersections.add( ((i, j), (k, l)) )
          elif (island_i['y'] == island_j['y'] and island_k['x'] == island_l['x']):
            # Горизонтальное ребро (i, j) и вертикальное ребро (k, l)
            if (min(island_i['x'], island_j['x']) < island_k['x'] < max(island_i['x'], island_j['x']) and
                min(island_k['y'], island_l['y']) < island_i['y'] < max(island_k['y'], island_l['y'])):
              intersections.add( ((i, j), (k, l)) )

      return intersections

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]  # Удаляем лишние пробелы в начале и конце строк
            width, height, _ = map(int, lines[0].split())
            grid = []
            for i in range(1, height + 1):
                row = list(map(int, lines[i].split()))
                grid.append(row)
            return Puzzle(width, height, grid)
