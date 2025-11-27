

### ЧАСТИНА 4: Структури даних (Дерева) та Теорія графів

#### 1. Дерево Фенвіка (Binary Indexed Tree)
**Пояснення:** Структура для швидкого знаходження префіксних сум і оновлення елементів. Обидві операції за $O(\log N)$. Компактніша і легша в написанні, ніж дерево відрізків.
Ключова операція: `i & -i` (отримання молодшого біта).

*   **Приклад 1: Оновлення (Add) та Префіксна сума**
    ```python
    n = 10
    bit = [0] * (n + 1) # 1-based indexing

    def update(i, delta):
        while i <= n:
            bit[i] += delta
            i += i & (-i)

    def query(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    update(3, 5) # додати 5 до A[3]
    update(5, 2) # додати 2 до A[5]
    print(query(5)) # сума A[1]...A[5] -> 7
    ```
*   **Приклад 2: Сума на відрізку [L, R]**
    ```python
    # Використовуємо функції з прикладу 1
    # Сума на [3, 5] = Sum(5) - Sum(2)
    l, r = 3, 5
    print(query(r) - query(l - 1)) 
    ```

#### 2. Дерево відрізків (Segment Tree)
**Пояснення:** Універсальна структура. Дозволяє виконувати будь-яку асоціативну операцію (сума, min, max, gcd) на відрізку і оновлювати елемент за $O(\log N)$. Зазвичай будується на масиві розміром $4N$.

*   **Приклад 1: Побудова (Build) для суми**
    ```python
    arr = [1, 2, 3, 4]
    n = len(arr)
    tree = [0] * (4 * n)

    def build(v, tl, tr):
        if tl == tr:
            tree[v] = arr[tl]
        else:
            tm = (tl + tr) // 2
            build(2*v, tl, tm)
            build(2*v+1, tm+1, tr)
            tree[v] = tree[2*v] + tree[2*v+1]

    build(1, 0, n-1)
    print(tree[1]) # Корінь (сума всього масиву) -> 10
    ```
*   **Приклад 2: Запит суми (Query)**
    ```python
    def sum_query(v, tl, tr, l, r):
        if l > r: return 0
        if l == tl and r == tr: return tree[v]
        tm = (tl + tr) // 2
        return sum_query(2*v, tl, tm, l, min(r, tm)) + \
               sum_query(2*v+1, tm+1, tr, max(l, tm+1), r)

    print(sum_query(1, 0, n-1, 1, 2)) # сума arr[1..2] (2+3) -> 5
    ```

#### 3. Пошук у ширину (BFS)
**Пояснення:** Обхід графа "шарами". Знаходить найкоротший шлях у **незваженому** графі. Використовує чергу.

*   **Приклад 1: Відстані від старту**
    ```python
    from collections import deque
    graph = {1: [2, 3], 2: [4], 3: [], 4: [1]}
    start = 1
    dist = {start: 0}
    q = deque([start])
    
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    print(dist) # {1: 0, 2: 1, 3: 1, 4: 2}
    ```
*   **Приклад 2: Лабіринт (сітка)**
    ```python
    # Знайти вихід 'E' зі старту 'S'
    grid = ["S..", ".#.", "..E"]
    R, C = 3, 3
    q = deque([(0, 0, 0)]) # r, c, dist
    visited = {(0,0)}
    
    while q:
        r, c, d = q.popleft()
        if grid[r][c] == 'E':
            print(f"Steps: {d}"); break
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc] != '#':
                visited.add((nr,nc))
                q.append((nr, nc, d+1))
    ```

#### 4. Пошук у довжину (DFS)
**Пояснення:** Обхід "в глибину". Використовується для пошуку циклів, перевірки зв'язності, топологічного сортування.

*   **Приклад 1: Пошук компонент зв'язності**
    ```python
    graph = {1: [2], 2: [1], 3: []}
    visited = set()
    
    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
    
    components = 0
    for node in graph:
        if node not in visited:
            dfs(node)
            components += 1
    print(components) # 2 (компоненти: {1,2} та {3})
    ```
*   **Приклад 2: Перевірка досяжності**
    ```python
    # Чи можна дійти з A в B?
    visited = set()
    def can_reach(u, target):
        if u == target: return True
        visited.add(u)
        for v in graph.get(u, []):
            if v not in visited:
                if can_reach(v, target): return True
        return False
    ```

#### 5. Алгоритм Дейкстри (Dijkstra)
**Пояснення:** Найкоротший шлях у графі з **невід'ємними** вагами. Використовує пріоритетну чергу (heap). Складність $O(E \log V)$.

*   **Приклад 1: Базова реалізація**
    ```python
    import heapq
    # (cost, neighbor)
    graph = {1: [(1, 2), (4, 3)], 2: [(2, 3)], 3: []} 
    start = 1
    
    pq = [(0, start)] # (distance, vertex)
    dists = {node: float('inf') for node in graph}
    dists[start] = 0
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dists[u]: continue # вже знайшли коротший шлях
        
        for weight, v in graph[u]:
            if dists[u] + weight < dists[v]:
                dists[v] = dists[u] + weight
                heapq.heappush(pq, (dists[v], v))
                
    print(dists) # {1: 0, 2: 1, 3: 3}
    ```
*   **Приклад 2: Відновлення шляху**
    ```python
    # Потрібно зберігати parent[v] = u, коли оновлюємо відстань
    parent = {start: None}
    # ... всередині циклу if ...
        parent[v] = u
    
    # Після алгоритму відновлюємо шлях з кінця
    curr = 3
    path = []
    while curr:
        path.append(curr)
        curr = parent.get(curr)
    print(path[::-1]) # [1, 2, 3]
    ```

#### 6. Алгоритм Флойда-Воршелла
**Пояснення:** Знаходить найкоротші відстані між **усіма парами** вершин. Працює за $O(V^3)$. Допускає від'ємні ваги (але без від'ємних циклів).

*   **Приклад 1: Класичні 3 цикли**
    ```python
    INF = float('inf')
    n = 3
    # Матриця суміжності
    dist = [[0, 5, INF],
            [INF, 0, 2],
            [INF, INF, 0]]
            
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                
    print(dist[0][2]) # 7 (0->1->2: 5+2)
    ```
*   **Приклад 2: Транзитивне замикання (Reachability)**
    *Чи можна взагалі дійти з i в j?*
    ```python
    # Замість min/+ використовуємо or/and
    reachable = [[True, True, False], [False, True, True], [False, False, True]]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reachable[i][j] = reachable[i][j] or (reachable[i][k] and reachable[k][j])
    ```

#### 7. Система неперетинних множин (СНМ / DSU)
**Пояснення:** Дозволяє об'єднувати множини та перевіряти, чи належать елементи до однієї множини. Ефективно для пошуку компонент зв'язності та алгоритму Крускала. Майже $O(1)$.

*   **Приклад 1: Реалізація з евристикою стиснення шляхів**
    ```python
    parent = list(range(10)) # кожен сам собі батько
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i]) # path compression
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j # simple union
            return True
        return False
        
    union(1, 2)
    print(find(1) == find(2)) # True
    ```
*   **Приклад 2: Підрахунок розміру компонент**
    ```python
    size = [1] * 10
    # В union:
    if root_i != root_j:
        if size[root_i] < size[root_j]: root_i, root_j = root_j, root_i
        parent[root_j] = root_i
        size[root_i] += size[root_j]
    ```

#### 8. Найменший спільний предок (LCA)
**Пояснення:** Знайти найнижчу вершину, яка є предком і для U, і для V. Найчастіше використовують метод **бінарного підйому** (Binary Lifting), щоб відповідати за $O(\log N)$.

*   **Приклад 1: Наївний метод (підняти на одну глибину)**
    *Тільки для розуміння або дуже малих дерев.*
    ```python
    # Припустимо, є масив parent і depth
    def get_lca_naive(u, v):
        if depth[u] < depth[v]: u, v = v, u # u глибше
        while depth[u] > depth[v]:
            u = parent[u]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    ```
*   **Приклад 2: Підготовка таблиці для бінарного підйому (ідея)**
    ```python
    LOG = 10 # 2^10 > N
    up = [[0] * LOG for _ in range(n)] 
    
    # up[u][i] - це 2^i-й предок вершини u
    # up[u][0] = parent[u]
    for j in range(1, LOG):
        for i in range(n):
            parent_halfway = up[i][j-1]
            up[i][j] = up[parent_halfway][j-1]
    ```

#### 9. Мости та шарніри
**Пояснення:**
*   **Міст:** Ребро, видалення якого збільшує кількість компонент зв'язності.
*   **Шарнір:** Вершина, видалення якої робить граф незв'язним.
Використовується DFS, час входу `tin` і функція `low` (найвищий предок, до якого можна дістатись зворотним ребром).

*   **Приклад 1: Логіка пошуку мостів (фрагмент DFS)**
    ```python
    timer = 0
    tin = [-1] * n
    low = [-1] * n
    
    def dfs_bridges(u, p=-1):
        global timer
        tin[u] = low[u] = timer; timer += 1
        for v in graph[u]:
            if v == p: continue
            if tin[v] != -1: # Back edge
                low[u] = min(low[u], tin[v])
            else:
                dfs_bridges(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    print(f"Bridge found: {u}-{v}")
    ```

---
**Чи можу я продовжувати з Частиною 5 (Складна математика, Рядки та Інші алгоритми)?**
