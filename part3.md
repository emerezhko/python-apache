Гаразд, переходимо до **Частини 3**. Це фундаментальні теми для олімпіад, особливо математика та бінарний пошук.

### ЧАСТИНА 3: Математика, Рекурсія, Бінарний пошук

#### 1. Евкліда алгоритм, НСД (GCD) та НСК (LCM)
**Пояснення:** НСД — найбільший спільний дільник. Алгоритм Евкліда базується на факті `gcd(a, b) = gcd(b, a % b)`. НСК обчислюється через НСД: `lcm(a, b) = (a * b) // gcd(a, b)`.

*   **Приклад 1: Власна реалізація та `math`**
    ```python
    import math

    # Вручну (рекурсивно або циклом)
    def gcd_manual(a, b):
        while b:
            a, b = b, a % b
        return a

    print(gcd_manual(48, 18)) # 6
    print(math.gcd(48, 18))   # 6 (швидше на олімпіадах)
    ```
*   **Приклад 2: Знаходження НСК (LCM)**
    ```python
    a, b = 12, 15
    # Python 3.9+ має math.lcm, але для старших версій:
    lcm_val = (a * b) // math.gcd(a, b)
    print(lcm_val) # 60
    ```

#### 2. Прості числа, перевірка на простоту
**Пояснення:** Число просте, якщо ділиться тільки на 1 і на себе. Перевіряти дільники достатньо до $\sqrt{N}$. Складність $O(\sqrt{N})$.

*   **Приклад 1: Оптимізована перевірка**
    ```python
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        # Перевіряємо тільки непарні до кореня
        d = 3
        while d * d <= n:
            if n % d == 0: return False
            d += 2
        return True

    print(is_prime(17)) # True
    ```
*   **Приклад 2: Розбиття на множники (факторизація)**
    ```python
    def factorize(n):
        factors = []
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1: # Якщо залишилось просте число
            factors.append(temp)
        return factors

    print(factorize(60)) # [2, 2, 3, 5]
    ```

#### 3. Решето Ератосфена
**Пояснення:** Алгоритм для знаходження *всіх* простих чисел до $N$. Набагато швидше, ніж перевіряти кожне число окремо. Складність $O(N \log \log N)$.

*   **Приклад 1: Класичне решето**
    ```python
    n = 30
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Починаємо з i*i, крок i
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
                
    primes = [x for x in range(n + 1) if is_prime[x]]
    print(primes) # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    ```
*   **Приклад 2: Лінійне решето (Min Prime Divisor) - просунутий варіант**
    *Корисне для факторизації багатьох чисел.*
    ```python
    N = 20
    lp = [0] * (N + 1) # найменший простий дільник
    pr = []
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            pr.append(i)
        for p in pr:
            if p > lp[i] or i * p > N: break
            lp[i * p] = p
            
    print(lp[10]) # 2 (бо 10 ділиться на 2)
    ```

#### 4. Рекурсія
**Пояснення:** Функція викликає сама себе. Важливо мати "базовий випадок" для зупинки.

*   **Приклад 1: Факторіал**
    ```python
    def factorial(n):
        if n == 0: return 1
        return n * factorial(n - 1)
    
    print(factorial(5)) # 120
    ```
*   **Приклад 2: Генерація всіх двійкових послідовностей довжини N**
    ```python
    def generate(n, current=""):
        if len(current) == n:
            print(current)
            return
        generate(n, current + "0")
        generate(n, current + "1")

    generate(2) 
    # Виведе:
    # 00
    # 01
    # 10
    # 11
    ```

#### 5. Бінарний пошук (Двійковий пошук)
**Пояснення:** Пошук елемента у *відсортованому* масиві діленням навпіл. Складність $O(\log N)$. Також використовується для пошуку відповіді (Binary Search on Answer).

*   **Приклад 1: Класичний бінарний пошук**
    ```python
    arr = [1, 3, 5, 7, 9, 11]
    target = 7
    
    low, high = 0, len(arr) - 1
    found_idx = -1
    
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            found_idx = mid
            break
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
            
    print(found_idx) # 3
    ```

*   **Приклад 2: Бінарний пошук по відповіді (Дуже важливо для олімпіад!)**
    *Задача: Знайти квадратний корінь цілого числа $N$ (цілу частину) без `sqrt`.*
    ```python
    n = 50
    low, high = 0, n
    ans = 0
    
    while low <= high:
        mid = (low + high) // 2
        if mid * mid <= n:
            ans = mid      # mid підходить, пробуємо більше
            low = mid + 1
        else:
            high = mid - 1 # забагато
            
    print(ans) # 7 (бо 7*7=49 <= 50, а 8*8=64 > 50)
    ```
    *Також у Python є модуль `bisect` (`bisect_left`, `bisect_right`) для роботи зі списками.*

---
**Чи можу я продовжувати з Частиною 4 (Графи та складніші структури даних - задачі F)?**
