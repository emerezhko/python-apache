

### ЧАСТИНА 1: Масиви, Рядки та Прості структури даних

#### 1. Масиви та рядки: Індексація
**Пояснення:** Доступ до елементів за їх порядковим номером (починається з 0). У Python підтримується від'ємна індексація (відлік з кінця).

*   **Приклад 1: Пряма та зворотна індексація**
    ```python
    arr = [10, 20, 30, 40, 50]
    print(arr[0])   # 10 (перший елемент)
    print(arr[-1])  # 50 (останній елемент)
    ```
*   **Приклад 2: Зрізи (Slicing)**
    ```python
    s = "olympiad"
    print(s[1:4])   # "lym" (з 1 по 3 включно)
    print(s[::-1])  # "daipmylo" (реверс рядка)
    ```

#### 2. Лінійний прохід (Max, Min, Sum, Пошук)
**Пояснення:** Перебір кожного елемента масиву один за одним (`O(N)`). Використовується для агрегації даних або пошуку.

*   **Приклад 1: Знаходження максимуму та суми вручну**
    ```python
    nums = [3, 1, 4, 1, 5, 9]
    mx = nums[0]
    total = 0
    for x in nums:
        if x > mx: mx = x
        total += x
    print(mx, total) # 9, 23
    ```
*   **Приклад 2: Лінійний пошук (чи є елемент)**
    ```python
    target = 5
    found = False
    for x in nums:
        if x == target:
            found = True
            break
    print("Found" if found else "Not Found")
    ```

#### 3. Пошук підмасивів та підрядків
**Пояснення:** Підмасив — це неперервна частина масиву. Перебір усіх підмасивів зазвичай займає `O(N^2)` або `O(N^3)`.

*   **Приклад 1: Виведення всіх підрядків**
    ```python
    s = "abc"
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            print(s[i : j+1]) 
    # Виведе: a, ab, abc, b, bc, c
    ```
*   **Приклад 2: Сума максимального підмасиву (наївна, O(N^2))**
    *Примітка: для O(N) існує алгоритм Кадане.*
    ```python
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    max_sum = float('-inf')
    for i in range(len(arr)):
        current_sum = 0
        for j in range(i, len(arr)):
            current_sum += arr[j]
            max_sum = max(max_sum, current_sum)
    print(max_sum) # 6 (це [4, -1, 2, 1])
    ```

#### 4. Квадратичні алгоритми сортування
**Пояснення:** Прості алгоритми (Bubble Sort, Insertion Sort, Selection Sort) зі складністю `O(N^2)`. На олімпіадах зазвичай краще використовувати вбудований `sort()` (який є `O(N log N)`), але розуміння принципу важливе.

*   **Приклад 1: Сортування бульбашкою (Bubble Sort)**
    ```python
    arr = [64, 34, 25, 12, 22]
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    print(arr)
    ```
*   **Приклад 2: Сортування вибором (Selection Sort)**
    ```python
    arr = [64, 25, 12, 22, 11]
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print(arr)
    ```

#### 5. Прості структури даних: Черга (Queue)
**Пояснення:** Принцип FIFO (First In, First Out — перший зайшов, перший вийшов). У Python ефективно використовується `collections.deque`.

*   **Приклад 1: Реалізація через deque**
    ```python
    from collections import deque
    q = deque()
    q.append(1) # push
    q.append(2)
    print(q.popleft()) # pop -> 1
    print(q[0])        # peek -> 2
    ```
*   **Приклад 2: Моделювання черги в магазині**
    ```python
    # Імена людей у черзі
    queue = deque(["Anna", "Bob", "Charlie"])
    while queue:
        person = queue.popleft()
        print(f"{person} обслужили")
    ```

#### 6. Прості структури даних: Стек (Stack)
**Пояснення:** Принцип LIFO (Last In, First Out — останній зайшов, перший вийшов). У Python використовується звичайний список `list`.

*   **Приклад 1: Базові операції**
    ```python
    stack = []
    stack.append(10) # push
    stack.append(20)
    print(stack.pop()) # pop -> 20
    print(stack[-1])   # top -> 10
    ```
*   **Приклад 2: Перевірка дужок (класична задача)**
    ```python
    s = "(()())"
    stack = []
    balanced = True
    for char in s:
        if char == '(':
            stack.append(char)
        elif stack: # якщо прийшла ')' і стек не пустий
            stack.pop()
        else:
            balanced = False; break
    if stack: balanced = False
    print(balanced)
    ```

#### 7. Прості структури даних: Множина (Set)
**Пояснення:** Колекція унікальних елементів без порядку. Операції пошуку/додавання в середньому `O(1)`.

*   **Приклад 1: Видалення дублікатів**
    ```python
    nums = [1, 2, 2, 3, 3, 3]
    unique = set(nums)
    print(unique) # {1, 2, 3}
    ```
*   **Приклад 2: Швидка перевірка наявності**
    ```python
    seen = {10, 20, 30}
    if 20 in seen: # O(1)
        print("Є в базі")
    seen.add(40)
    ```

#### 8. Прості структури даних: Асоціативний масив (Dictionary)
**Пояснення:** Зберігає пари "ключ-значення". Доступ за ключем миттєвий `O(1)`.

*   **Приклад 1: Підрахунок частоти символів**
    ```python
    s = "hello"
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    print(freq) # {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    ```
*   **Приклад 2: Зберігання графа (список суміжності)**
    ```python
    graph = {
        1: [2, 3],
        2: [1, 4],
        3: [1],
        4: [2]
    }
    print(graph[1]) # сусіди вершини 1 -> [2, 3]
    ```
