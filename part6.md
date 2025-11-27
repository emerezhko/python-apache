*Важливо для Python на цьому рівні:* Завжди збільшуйте ліміт рекурсії (`import sys; sys.setrecursionlimit(200000)`), використовуйте `sys.stdin.readline` для швидкого вводу.

### ЧАСТИНА 6: Експертний рівень (Задачі G)

#### 1. Sparse Table (Розріджена таблиця)
**Пояснення:** Статична структура даних для запитів на відрізку (RMQ — Range Minimum Query) за $O(1)$. Будується за $O(N \log N)$. Працює для ідемпотентних операцій (min, max, gcd), але не для суми.

*   **Приклад 1: Побудова таблиці (Precomputation)**
    ```python
    import math
    arr = [1, 3, 4, -1, 7, 8]
    n = len(arr)
    k = int(math.log2(n)) + 1
    st = [[0] * k for _ in range(n)]

    # База: інтервали довжиною 1
    for i in range(n):
        st[i][0] = arr[i]

    # Динаміка: st[i][j] покриває [i, i + 2^j - 1]
    for j in range(1, k):
        for i in range(n - (1 << j) + 1):
            st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1])
    ```
*   **Приклад 2: Відповідь на запит за O(1)**
    ```python
    def query(L, R): # [L, R] включно
        length = R - L + 1
        j = int(math.log2(length))
        # Мінімум з двох перекриваючихся відрізків довжиною 2^j
        return min(st[L][j], st[R - (1 << j) + 1][j])

    print(query(1, 4)) # min(3, 4, -1, 7) -> -1
    ```

#### 2. Коренева декомпозиція (Sqrt Decomposition)
**Пояснення:** Розбиття масиву на блоки розміром $\sqrt{N}$. Дозволяє обробляти запити (сума, модифікація) за $O(\sqrt{N})$. Гнучкіша за дерево відрізків (алгоритм Мо).

*   **Приклад 1: Point Update / Range Sum**
    ```python
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = len(arr)
    sq = int(n**0.5) + 1
    blocks = [0] * sq
    
    # Побудова
    for i in range(n):
        blocks[i // sq] += arr[i]
        
    # Запит суми [l, r]
    def query_sqrt(l, r):
        res = 0
        # "Лівий хвіст"
        while l <= r and l % sq != 0:
            res += arr[l]; l += 1
        # Цілі блоки
        while l + sq - 1 <= r:
            res += blocks[l // sq]; l += sq
        # "Правий хвіст"
        while l <= r:
            res += arr[l]; l += 1
        return res
    ```
*   **Приклад 2: Сортування запитів для Алгоритму Мо**
    ```python
    # Сортуємо запити так, щоб мінімізувати рух вказівників
    # Key: (block_index, r)
    sq = 320 
    queries = [(1, 5, 0), (2, 8, 1)] # (l, r, id)
    queries.sort(key=lambda x: (x[0] // sq, x[1] if (x[0]//sq)%2==0 else -x[1]))
    ```

#### 3. Алгоритм Крускала (MST — Мінімальне кістякове дерево)
**Пояснення:** Жадібний алгоритм. Сортуємо ребра за вагою і додаємо ребро, якщо воно не утворює циклу (використовуємо DSU).

*   **Приклад 1: Основний цикл**
    ```python
    # edges = [(weight, u, v), ...]
    # parent = [...] (ініціалізований DSU)
    
    edges.sort()
    mst_weight = 0
    edges_count = 0
    
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_weight += w
            edges_count += 1
    
    print(mst_weight)
    ```
*   **Приклад 2: DSU для Крускала (лаконічно)**
    ```python
    parent = list(range(n + 1))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False
    ```

#### 4. Декартове дерево (Treap: Tree + Heap)
**Пояснення:** Бінарне дерево пошуку за ключем $X$ і куча за пріоритетом $Y$ (пріоритети випадкові). Забезпечує балансування $O(\log N)$. Основні операції: `split` (розрізання) і `merge` (злиття).

*   **Приклад 1: Структура вузла**
    ```python
    import random
    class Node:
        def __init__(self, key):
            self.key = key
            self.priority = random.random()
            self.left = None
            self.right = None
    ```
*   **Приклад 2: Логіка Split (розбиття за ключем)**
    ```python
    def split(root, key):
        if not root: return None, None
        if root.key <= key:
            # Всі зліва йдуть в перше дерево, корінь теж
            r1, r2 = split(root.right, key)
            root.right = r1
            return root, r2
        else:
            l1, l2 = split(root.left, key)
            root.left = l2
            return l1, root
    ```

#### 5. Ейлеровий шлях / цикл
**Пояснення:** Прохід по всіх ребрах графа рівно один раз.
Умова існування циклу: граф зв'язний, всі вершини мають парну степінь.
Умова існування шляху: 0 або 2 вершини мають непарну степінь.

*   **Приклад 1: Алгоритм Гірхольцера (Hierholzer)**
    ```python
    # graph - список суміжності (використовуємо як стек/pop)
    stack = [start_node]
    path = []
    
    while stack:
        u = stack[-1]
        if graph[u]:
            v = graph[u].pop() # Видаляємо ребро
            # Для неорієнтованого треба видалити і u з graph[v]
            graph[v].remove(u) 
            stack.append(v)
        else:
            path.append(stack.pop())
    print(path[::-1]) # Результат треба перевернути
    ```
*   **Приклад 2: Перевірка ступенів**
    ```python
    degrees = [0] * n
    # ... підрахунок ...
    odd_count = sum(1 for d in degrees if d % 2 != 0)
    if odd_count == 0: print("Eulerian Cycle")
    elif odd_count == 2: print("Eulerian Path")
    else: print("None")
    ```

#### 6. Компоненти сильної зв'язності (SCC)
**Пояснення:** В орієнтованому графі це групи вершин, де з будь-якої можна дійти до будь-якої іншої в межах групи. Алгоритм Косарайю (2 DFS) або Тар'яна.

*   **Приклад 1: Алгоритм Косарайю (Крок 1 - порядок виходу)**
    ```python
    order = []
    visited = [False] * n
    def dfs1(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]: dfs1(v)
        order.append(u) # запам'ятовуємо час виходу
    
    for i in range(n):
        if not visited[i]: dfs1(i)
    ```
*   **Приклад 2: Алгоритм Косарайю (Крок 2 - DFS на оберненому графі)**
    ```python
    # adj_rev - граф з розвернутими ребрами
    visited = [False] * n
    component = []
    def dfs2(u):
        visited[u] = True
        component.append(u)
        for v in adj_rev[u]:
            if not visited[v]: dfs2(v)

    # Йдемо у зворотному порядку order
    while order:
        u = order.pop()
        if not visited[u]:
            component = []
            dfs2(u)
            print(f"SCC: {component}")
    ```

#### 7. Тернарний пошук (Ternary Search)
**Пояснення:** Пошук мінімуму/максимуму **унімодальної** функції (яка спочатку зростає, потім спадає, або навпаки). Розбиваємо відрізок на 3 частини точками `m1` і `m2`.

*   **Приклад 1: Неперервна функція**
    ```python
    def f(x): return (x - 2)**2 + 3 # мінімум в x=2
    
    l, r = -10, 10
    for _ in range(100): # 100 ітерацій для точності
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if f(m1) < f(m2):
            r = m2
        else:
            l = m1
    print(l) # ~2.0
    ```
*   **Приклад 2: Дискретний випадок (пошук в масиві)**
    ```python
    # Поки відстань > 2, звужуємо межі
    while r - l > 2:
        m1 = l + (r - l) // 3
        m2 = r - (r - l) // 3
        if arr[m1] < arr[m2]: r = m2
        else: l = m1
    # В кінці перевіряємо невеликий діапазон [l, r] вручну
    print(min(arr[l:r+1]))
    ```

#### 8. Бітсети (Bitsets)
**Пояснення:** Використання довгих чисел як масивів булевих значень. Операції `&`, `|`, `^`, `<<` працюють у 32/64 рази швидше, ніж цикли.

*   **Приклад 1: Задача про рюкзак (досяжність суми)**
    ```python
    # Чи можна набрати вагу W, маючи предмети items?
    dp = 1 # бітова маска, 1 на 0-й позиції означає суму 0
    items = [2, 5, 3]
    for x in items:
        dp |= (dp << x) # зсув вліво = додавання x до всіх існуючих сум
        
    target = 7
    if (dp >> target) & 1:
        print("Possible")
    ```
*   **Приклад 2: Спільні друзі (швидкий перетин множин)**
    ```python
    # users[i] - бітова маска друзів i-го юзера
    user1 = 0b10110
    user2 = 0b01110
    common = user1 & user2
    print(bin(common).count('1')) # Кількість спільних
    ```

#### 9. Суфіксний масив (Suffix Array)
**Пояснення:** Масив індексів початку суфіксів, відсортованих лексикографічно. Дозволяє шукати підрядки бінарним пошуком, знаходити найбільший спільний префікс (LCP) тощо.

*   **Приклад 1: Наївна побудова (O(N^2 log N))**
    *На олімпіадах для N < 1000-5000 працює, для більших треба O(N log N) алгоритм, який складний у реалізації.*
    ```python
    s = "banana"
    suffixes = []
    for i in range(len(s)):
        suffixes.append((s[i:], i))
    suffixes.sort()
    sa = [idx for suffix, idx in suffixes]
    print(sa) # [5, 3, 1, 0, 4, 2] -> a, ana, anana, banana, na, nana
    ```
*   **Приклад 2: Пошук підрядка через SA (ідея)**
    ```python
    # Якщо маємо SA, можемо використовувати bisect для пошуку P в списку суфіксів.
    # Це дає O(|P| * log N)
    ```

---
### Загальні поради для олімпіади

1.  **Починайте з простого:** Якщо задача здається складною, спробуйте "повний перебір" (Brute Force). Якщо N маленьке ($N \le 20$), це спрацює.
2.  **Оцінюйте складність:**
    *   $N \le 10^6 \to O(N)$ або $O(N \log N)$
    *   $N \le 10^5 \to O(N \log N)$ або $O(N \sqrt{N})$
    *   $N \le 5000 \to O(N^2)$
    *   $N \le 500 \to O(N^3)$ (Флойд)
3.  **Тести:** Завжди перевіряйте граничні випадки: $N=0, N=1$, всі числа однакові, граф — лінія, від'ємні числа.

Успіхів на олімпіаді!
