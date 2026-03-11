# ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №1
## Тема: "Основы линейной алгебры: Векторы и матрицы в Python"

**Выполнил:** Студент(ка) 3 курса Смородин А.А. 
**Группа:** 1 группа 2 подгруппа

---

## 1. ЦЕЛЬ РАБОТЫ

Изучить основы работы с векторами и матрицами в Python с использованием библиотеки NumPy и понять их применение в задачах искусственного интеллекта.


---

## 2. ИМПОРТ БИБЛИОТЕК

### Код программы:

```python
# ИМПОРТ БИБЛИОТЕК
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Настройка для красивого отображения графиков
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("Библиотеки успешно импортированы!")
print(f"Версия NumPy: {np.__version__}")
```
Результата работы 

![результат работы](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202026-03-11%20153338.png)
## 3. РАБОТА С ВЕКТОРАМИ
### 3.1 Создание векторов

### Код программы:

```python
# ========== ЧАСТЬ 1.1: Создание векторов ==========
print("\n" + "="*50)
print("ЧАСТЬ 1.1: Создание векторов")
print("="*50)

# Создаем векторы
a = np.array([3, 4])
b = np.array([1, 2])
c = np.array([2, -1, 3])

print(f"Вектор a: {a}")
print(f"Вектор b: {b}")
print(f"Вектор c: {c}")
print(f"Размерность вектора a: {a.shape}")
print(f"Размерность вектора c: {c.shape}")
```
Результат работы 

![Создание векторов](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2.png)
### 3.2 Функция для визуализации векторов

### Код программы:

```python
# Функция для рисования векторов на плоскости
def plot_2d_vectors(*vectors, labels=None, colors=None):
    """Функция для визуализации 2D векторов"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    if labels is None:
        labels = [f'v{i+1}' for i in range(len(vectors))]
    
    for i, vector in enumerate(vectors):
        ax.arrow(0, 0, vector[0], vector[1], 
                head_width=0.2, head_length=0.2, 
                fc=colors[i % len(colors)], ec=colors[i % len(colors)],
                linewidth=2, label=labels[i])
    
    max_val = max([max(abs(v)) for v in vectors]) + 1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title('Визуализация векторов')
    plt.show()

# Рисуем векторы a и b
plot_2d_vectors(a, b, labels=['a = [3, 4]', 'b = [1, 2]'])
```
Результат работы 

![визуализация векторов](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%92%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2.png)
### 3.3 Сложение векторов

### Код программы:

```python
# ========== ЧАСТЬ 1.2: Сложение векторов ==========
print("\n" + "="*50)
print("ЧАСТЬ 1.2: Сложение векторов")
print("="*50)

sum_ab = a + b
print(f"a + b = {sum_ab}")

# Рисуем результат сложения
plot_2d_vectors(a, b, sum_ab, 
                labels=['a', 'b', 'a + b'], 
                colors=['red', 'blue', 'green'])
```
Результат работы 

![сложение векторов визуализация](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2.png)
![Сложение векторов консоль](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%BE%D0%B2%202.png)
### 3.4 Умножение вектора на скаляр

### Код программы:

```python
# ========== ЧАСТЬ 1.3: Умножение на скаляр ==========
print("\n" + "="*50)
print("ЧАСТЬ 1.3: Умножение на скаляр")
print("="*50)

scaled_a = 2 * a
print(f"2 * a = {scaled_a}")

# Рисуем результат умножения
plot_2d_vectors(a, scaled_a, 
                labels=['a', '2*a'], 
                colors=['red', 'orange'])
```
Результат работы 

![умножение на скаляр визуализаця](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A3%D0%BC%D0%BD%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%20%D1%81%D0%BA%D0%B0%D0%BB%D1%8F%D1%80.png)
![умножение на скаляр консоль](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A3%D0%BC%D0%BD%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BD%D0%B0%20%D1%81%D0%BA%D0%B0%D0%BB%D1%8F%D1%80%202.png)
### 3.5 Скалярное произведение и норма векторов

### Код программы:

```python
# ========== ЧАСТЬ 1.4: Скалярное произведение и норма ==========
print("\n" + "="*50)
print("ЧАСТЬ 1.4: Скалярное произведение и норма")
print("="*50)

dot_product = np.dot(a, b)
print(f"Скалярное произведение a · b = {dot_product}")

norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f"Норма вектора a: ||a|| = {norm_a:.3f}")
print(f"Норма вектора b: ||b|| = {norm_b:.3f}")

cos_angle = dot_product / (norm_a * norm_b)
angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)
print(f"Косинус угла между a и b: {cos_angle:.3f}")
print(f"Угол между векторами: {angle_deg:.1f}°")
```
Результат работы 

![скалярное произведение](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BA%D0%B0%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D0%B5%20%D0%BF%D1%80%D0%BE%D0%B8%D0%B7%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%BD%D0%BE%D1%80%D0%BC%D0%B0.png)
## 4. РАБОТА С МАТРИЦАМИ
### 4.1 Создание матриц

### Код программы:

```python
# ========== ЧАСТЬ 2.1: Создание матриц ==========
print("\n" + "="*50)
print("ЧАСТЬ 2.1: Создание матриц")
print("="*50)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[1, 2, 3], [4, 5, 6]])

print("Матрица A:")
print(A)
print(f"Размерность A: {A.shape}")

print("\nМатрица B:")
print(B)
print(f"Размерность B: {B.shape}")

print("\nМатрица C:")
print(C)
print(f"Размерность C: {C.shape}")
```
Результат работы 

![создание матриц](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86.png)

## 4.2 Операции над матрицами

### Код программы:

```python
# ========== ЧАСТЬ 2.2: Операции с матрицами ==========
print("\n" + "="*50)
print("ЧАСТЬ 2.2: Операции с матрицами")
print("="*50)

# TODO 2.2: Сложение матриц
sum_AB = A + B

print("A + B =")
print(sum_AB)

# TODO 2.3: Умножение матрицы на скаляр
scaled_A = 3 * A

print("\n3 * A =")
print(scaled_A)

# TODO 2.4: Умножение матриц
# ВНИМАНИЕ: умножение матриц - это НЕ поэлементное умножение!
product_AB = np.dot(A, B)  # или A @ B

print("\nA * B =")
print(product_AB)

# Проверяем, что умножение матриц не коммутативно (A*B ≠ B*A)
product_BA = np.dot(B, A)
print("\nB * A =")
print(product_BA)

print(f"\nA*B == B*A? {np.array_equal(product_AB, product_BA)}")
# ========== ЧАСТЬ 2.3: Умножение матрицы на вектор ==========
print("\n" + "="*50)
print("ЧАСТЬ 2.3: Умножение матрицы на вектор")
print("="*50)

# TODO 2.5: Умножаем матрицу A на вектор a
# Вспомним: a = [3, 4]
result_Aa = A @ a  # или np.dot(A, a)

print(f"A * a = {result_Aa}")

print("\nГеометрическая интерпретация:")
print(f"Исходный вектор a: {a}")
print(f"Преобразованный вектор A*a: {result_Aa}")

# Визуализируем преобразование
plot_2d_vectors(a, result_Aa,
                labels=['Исходный вектор a', 'Преобразованный A*a'],
                colors=['blue', 'red'])
```
Результат работы 

![Операции над матрицами](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%86%D0%B8%D0%B8%20%D1%81%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0%D0%BC%D0%B8.png)
![умножение матрицы на вектор](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A3%D0%BC%D0%BD%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D1%8B%20%D0%BD%D0%B0%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80.png)
![визуализация умножения матрицы на вектор](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A3%D0%BC%D0%BD%D0%BE%D0%B6%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B0%D1%82%D1%80%D1%86%D0%B8%D1%86%D1%8B%20%D0%BD%D0%B0%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80.png)
## 4.3 Транспонирование и специальные матрицы
### Код программы

```python
# ========== ЧАСТЬ 2.4: Транспонирование ==========
print("\n" + "="*50)
print("ЧАСТЬ 2.4: Транспонирование")
print("="*50)

A_T = A.T
print("Матрица A:")
print(A)
print("\nТранспонированная матрица A^T:")
print(A_T)

print("\nЕдиничная матрица 3x3:")
I = np.eye(3)
print(I)

print("\nНулевая матрица 2x3:")
zeros = np.zeros((2, 3))
print(zeros)

print("\nМатрица из единиц 2x2:")
ones = np.ones((2, 2))
print(ones)
```
Результат работы 

![транспонирование матриц](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%A2%D1%80%D0%B0%D0%BD%D1%81%D0%BF%D0%BE%D0%BD%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86.png)

## 5. ПРИМЕНЕНИЕ В ЗАДАЧАХ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА
### 5.1 Представление изображения как матрицы
### Код программы

```python
# ========== ЧАСТЬ 3.1: Изображение как матрица ==========
print("\n" + "="*50)
print("ЧАСТЬ 3.1: Изображение как матрица")
print("="*50)

smiley = np.array([
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0]
])

print("Изображение как матрица (первые 4 строки):")
print(smiley[:4])
print(f"Размерность: {smiley.shape}")
print(f"Всего пикселей: {smiley.size}")

plt.figure(figsize=(6, 6))
plt.imshow(smiley, cmap='gray', interpolation='nearest')
plt.title('Смайлик как матрица')
plt.colorbar(label='Яркость пикселя')
plt.show()

smiley_vector = smiley.flatten()
print(f"\nИзображение как вектор (первые 10 элементов):")
print(smiley_vector[:10])
print(f"Размерность вектора: {smiley_vector.shape}")
```
Результат работы

![Изображение как матрица](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BA%D0%B0%D0%BA%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D0%B0.png)
![изображение как вектор](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BA%D0%B0%D0%BA%20%D0%B2%D0%B5%D0%BA%D1%82%D0%BE%D1%80%20.png)
![матрица как смайлик](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%98%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86.png)

### 5.2 Косинусная мера сходства для текстов
### Код програмы:

```python
# ========== ЧАСТЬ 3.2: Косинусная мера для текстов ==========
print("\n" + "="*50)
print("ЧАСТЬ 3.2: Косинусная мера для текстов")
print("="*50)

# Словарь слов: ["кот", "собака", "компьютер", "алгоритм", "учиться", "дом", "машина"]
doc1 = np.array([3, 1, 0, 0, 1, 2, 0])  # про животных и дом
doc2 = np.array([2, 2, 0, 0, 1, 1, 0])  # тоже про животных
doc3 = np.array([0, 0, 3, 2, 1, 0, 1])  # про технологии
doc4 = np.array([1, 1, 1, 1, 2, 0, 1])  # смешанный

documents = [doc1, doc2, doc3, doc4]
doc_names = ["Документ 1 (животные)", "Документ 2 (животные)", 
             "Документ 3 (технологии)", "Документ 4 (смешанный)"]

print("Векторные представления документов:")
for i, (doc, name) in enumerate(zip(documents, doc_names)):
    print(f"{name}: {doc}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

n_docs = len(documents)
similarity_matrix = np.zeros((n_docs, n_docs))

for i in range(n_docs):
    for j in range(n_docs):
        similarity_matrix[i, j] = cosine_similarity(documents[i], documents[j])

print("\nМатрица косинусного сходства:")
print(similarity_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Косинусное сходство')
plt.xticks(range(n_docs), ['Док1', 'Док2', 'Док3', 'Док4'])
plt.yticks(range(n_docs), ['Док1', 'Док2', 'Док3', 'Док4'])
plt.title('Матрица косинусного сходства документов')

for i in range(n_docs):
    for j in range(n_docs):
        plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                ha='center', va='center', fontweight='bold')
plt.show()
```
Результат работы

![косинусная мера консоль](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%9A%D0%BE%D1%81%D0%B8%D0%BD%D1%83%D1%81%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%B5%D1%80%D0%B0%20%D0%B4%D0%BB%D1%8F%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2%202.png)
![косинусная мера визуализация](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%9A%D0%BE%D1%81%D0%B8%D0%BD%D1%83%D1%81%D0%BD%D0%B0%D1%8F%20%D0%BC%D0%B5%D1%80%D0%B0%20%D0%B4%D0%BB%D1%8F%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2.png)

### 5.3 Линейная регрессия
### Код программы:

```python
# ========== ЧАСТЬ 3.3: Линейная регрессия ==========
print("\n" + "="*50)
print("ЧАСТЬ 3.3: Линейная регрессия")
print("="*50)

# Данные: площадь дома и цена
areas = np.array([50, 60, 70, 80, 90, 100, 110, 120])  # площадь в кв.м
prices = np.array([3.2, 3.8, 4.1, 4.7, 5.2, 5.8, 6.1, 6.9])  # цена в млн руб.

print("Исходные данные:")
print(f"Площади: {areas}")
print(f"Цены: {prices}")

# Создаем матрицу признаков (добавляем столбец единиц для свободного члена)
X = np.column_stack([np.ones(len(areas)), areas])
y = prices

print("\nМатрица признаков X (первые 5 строк):")
print(X[:5])
print(f"\nВектор целевых значений y: {y}")

# Решаем нормальное уравнение: w = (X^T * X)^(-1) * X^T * y
X_T = X.T
XTX = X_T @ X
XTX_inv = np.linalg.inv(XTX)
XTy = X_T @ y
w = XTX_inv @ XTy

print(f"\nВеса модели: w = {w}")
print(f"Свободный член (bias): {w[0]:.3f}")
print(f"Коэффициент при площади: {w[1]:.3f}")
print(f"Формула: цена = {w[0]:.3f} + {w[1]:.3f} × площадь")

# Предсказания
predictions = X @ w

plt.figure(figsize=(10, 6))
plt.scatter(areas, prices, color='blue', label='Реальные данные', s=50)
plt.plot(areas, predictions, color='red', label='Модель', linewidth=2)
plt.xlabel('Площадь дома (кв.м)')
plt.ylabel('Цена (млн руб.)')
plt.title('Линейная регрессия: Предсказание цены дома')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Предсказание для нового дома
new_area = 95
new_house = np.array([1, new_area])
predicted_price = new_house @ w
print(f"\nПредсказанная цена дома {new_area} кв.м: {predicted_price:.2f} млн руб.")
```

Результат работы 

![линейная регрессия консоль](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%202.png)
![линейная регрессия визуал](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20.png)

## 6. ДОПОЛНИТЕЛЬНЫЕ ЗАДАНИЯ
### Код программы:
```python
# ========== ДОПОЛНИТЕЛЬНЫЕ ЗАДАНИЯ ==========
print("\n" + "="*50)
print("ДОПОЛНИТЕЛЬНЫЕ ЗАДАНИЯ")
print("="*50)

# Задание 3: Матрица поворота
def rotation_matrix(angle_degrees):
    """Создает матрицу поворота для 2D"""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array([[cos_a, -sin_a], 
                     [sin_a, cos_a]])

# Поворачиваем вектор a на 45 градусов
angle = 45
R = rotation_matrix(angle)
a_rotated = R @ a

print(f"\nЗадание 3: Поворот на {angle}°")
print(f"Исходный вектор a: {a}")
print(f"После поворота: {a_rotated}")
print(f"Длина не изменилась: {np.linalg.norm(a):.3f} -> {np.linalg.norm(a_rotated):.3f}")

plot_2d_vectors(a, a_rotated, 
                labels=[f'Исходный a', f'Повернутый на {angle}°'], 
                colors=['blue', 'green'])
```

Результат работы 
![Доп задание консоль](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%94%D0%BE%D0%BF%20%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B4%D0%BB%D1%8F%20%D1%81%D0%B0%D0%BC%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D1%82%D0%B5%D1%8C%D0%BD%D0%BE%D0%B9%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%202.png)
![Доп задание](https://github.com/Vox0n/Basic-Mathematics-for-IA/blob/main/LB1/picture/%D0%94%D0%BE%D0%BF%20%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B4%D0%BB%D1%8F%20%D1%81%D0%B0%D0%BC%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D1%82%D0%B5%D1%8C%D0%BD%D0%BE%D0%B9%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B.png)



