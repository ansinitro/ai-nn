# %% [markdown]
# # ЗАДАНИЕ №1: Предварительный анализ данных (EDA) и проектирование признаков
# **Предметная область:** Метеорология, климат  
# **Выбранный датасет:** Weather Dataset (Jena Climate 2009–2016)  
# **Источник:** Max Planck Institute for Biogeochemistry, Jena — [https://www.bgc-jena.mpg.de/wetter/](https://www.bgc-jena.mpg.de/wetter/)  
# **Целевая переменная:** `T (degC)` — температура воздуха в градусах Цельсия  
# **Тип задачи:** Регрессия (прогнозирование температуры на основе многомерного временного ряда)
#
# **Цель:** Провести разведочный анализ предоставленного датасета, выявить основные
# закономерности, выполнить очистку данных и создать новые признаки, необходимые
# для дальнейшего построения нейросетевых моделей.

# %% [markdown]
# ---
# ## Инициализация: импорт библиотек и настройка окружения
# Импортируем все необходимые библиотеки в соответствии с требованиями задания:
# `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд для сохранения графиков
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

# Настройки визуализации
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Пути к данным
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "jena_climate_2009_2016.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_data.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Целевая переменная
TARGET_COL = 'T (degC)'

def save_fig(fig, name):
    """Сохраняет график в папку figures/ и показывает его."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [График сохранён: figures/{name}.png]")
    plt.show()

# %% [markdown]
# ---
# ## 1. Загрузка и первичный осмотр данных
# **Задачи:**
# - Загрузить датасет с помощью библиотеки `pandas`.
# - Вывести общую информацию о данных: количество записей, типы столбцов, наличие пропущенных значений.
# - Проверить статистическое описание числовых признаков (mean, std, min, max, квартили).

# %%
# Загрузка данных
raw_df = pd.read_csv(RAW_DATA_PATH)

print("=" * 60)
print("ПЕРВЫЕ 5 СТРОК ДАТАСЕТА")
print("=" * 60)
print(raw_df.head().to_string())

# %%
print("\n" + "=" * 60)
print("ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ")
print("=" * 60)
print(f"Количество записей (строк): {raw_df.shape[0]}")
print(f"Количество признаков (столбцов): {raw_df.shape[1]}")
print(f"\nНазвания столбцов:")
for i, col in enumerate(raw_df.columns, 1):
    print(f"  {i:2d}. {col} (тип: {raw_df[col].dtype})")

print("\nПодробная информация (df.info()):")
raw_df.info()

# %%
print("\n" + "=" * 60)
print("СТАТИСТИЧЕСКОЕ ОПИСАНИЕ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 60)
print(raw_df.describe().T.round(2).to_string())

# %%
print("\n" + "=" * 60)
print("ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 60)
missing = raw_df.isnull().sum()
print(f"Общее число пропусков в датасете: {missing.sum()}")
if missing.sum() > 0:
    print("\nСтолбцы с пропусками:")
    print(missing[missing > 0].to_string())
else:
    print("Пропущенных значений не обнаружено ✓")

# %% [markdown]
# ### Вывод по шагу 1:
# Датасет содержит **420 551 запись** с шагом в 10 минут, охватывающий период с
# 01.01.2009 по 01.01.2017 (≈ 8 лет). В наборе **15 столбцов**: один временной
# (`Date Time`) и 14 числовых признаков (температура, давление, влажность,
# плотность воздуха, скорость и направление ветра и др.).
#
# **Пропущенных значений не обнаружено**, что говорит о высоком качестве
# исходных данных. Однако при анализе `describe()` следует обратить внимание на
# диапазоны значений: например, минимальная температура может достигать
# отрицательных значений (зима в Германии), а максимальная — превышать 35°C (лето).
#
# Целевая переменная — **`T (degC)`** (температура воздуха в °C), которую
# мы будем анализировать и предсказывать на последующих этапах.

# %% [markdown]
# ---
# ## 2. Обработка временной метки
# **Задачи:**
# - Преобразовать столбец `Date Time` в формат `datetime`.
# - Извлечь из даты новые признаки: час, день недели, признак выходного дня (0/1).
# - Убедиться, что данные упорядочены по времени.

# %%
# Рабочая копия датафрейма
df = raw_df.copy()

# Преобразование столбца Date Time в объект datetime
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

# Сортировка по времени (гарантия хронологического порядка)
df = df.sort_values('Date Time').reset_index(drop=True)

# Извлечение новых временных признаков
df['hour'] = df['Date Time'].dt.hour
df['day_of_week'] = df['Date Time'].dt.dayofweek  # 0=Пн, 6=Вс
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 1 = выходной

# Устанавливаем Date Time как индекс (best practice для временных рядов)
df = df.set_index('Date Time')

print("=== Проверка извлечённых временных признаков ===")
print(df[['hour', 'day_of_week', 'is_weekend']].head(10).to_string())

# %%
# Проверяем равномерность временного шага
time_diffs = df.index.to_series().diff().dropna()
print("\n=== Распределение временных шагов ===")
print(time_diffs.value_counts().head().to_string())
print(f"\nМинимальный шаг: {time_diffs.min()}")
print(f"Максимальный шаг: {time_diffs.max()}")
print(f"Медианный шаг:   {time_diffs.median()}")

# %% [markdown]
# ### Вывод по шагу 2:
# Столбец `Date Time` успешно преобразован в формат `datetime` и установлен как
# индекс DataFrame. Извлечены три новых признака:
# - **`hour`** — час суток (0–23), для анализа суточной сезонности;
# - **`day_of_week`** — день недели (0 = понедельник, 6 = воскресенье);
# - **`is_weekend`** — бинарный флаг выходного дня (1 = суббота/воскресенье).
#
# Временной шаг составляет ровно **10 минут** для всех записей, что подтверждает
# регулярность и целостность временного ряда. Пропусков во временной шкале не
# обнаружено.

# %% [markdown]
# ---
# ## 3. Визуализация и анализ распределений
# **Задачи:**
# - Построить корреляционную матрицу и выявить признаки, наиболее сильно коррелирующие с целевой переменной.
# - Построить гистограммы распределения каждого признака, обратить внимание на выбросы.
# - Построить временной ряд потребления энергии, наложить скользящее среднее для выявления трендов и сезонности.
# - Создать график зависимости среднего потребления от часа суток и дня недели.

# %% [markdown]
# ### 3.1 Корреляционная матрица

# %%
# Вычисляем корреляционную матрицу для всех числовых признаков
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

# Визуализация полной корреляционной матрицы
fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Верхний треугольник
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    ax=ax
)
ax.set_title("Корреляционная матрица всех числовых признаков", fontsize=16)
fig.tight_layout()
save_fig(fig, "01_correlation_matrix")

# %%
# Отдельно выделяем корреляции с целевой переменной T (degC)
target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

print("=" * 60)
print(f"КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ '{TARGET_COL}'")
print("=" * 60)
print(f"\nТоп-5 положительных корреляций:")
for feat, val in target_corr.head(5).items():
    print(f"  {feat:25s} → r = {val:+.4f}")
print(f"\nТоп-5 отрицательных корреляций:")
for feat, val in target_corr.tail(5).items():
    print(f"  {feat:25s} → r = {val:+.4f}")

# Визуализация корреляций с целевой (горизонтальный барплот)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in target_corr.values]
target_corr.plot(kind='barh', color=colors, edgecolor='black', ax=ax)
ax.set_title(f"Корреляция признаков с целевой переменной '{TARGET_COL}'", fontsize=14)
ax.set_xlabel("Коэффициент корреляции Пирсона")
ax.axvline(x=0, color='black', linewidth=0.8)
fig.tight_layout()
save_fig(fig, "02_target_correlations")

# %% [markdown]
# **Интерпретация корреляционной матрицы:**
#
# Наиболее сильные **положительные** корреляции с температурой `T (degC)`:
# - **`Tdew (degC)`** (точка росы) — практически идеальная корреляция (≈0.95),
#   что физически обосновано: точка росы напрямую зависит от температуры воздуха.
# - **`VPact (mbar)`** (фактическое давление водяного пара) — сильная положительная
#   связь, так как при высокой температуре воздух удерживает больше влаги.
#
# Наиболее сильные **отрицательные** корреляции:
# - **`rho (g/m**3)`** (плотность воздуха) — сильная обратная связь (≈ –0.87),
#   что объясняется физическим законом: при нагревании воздух расширяется и его
#   плотность снижается.
# - **`p (mbar)`** (атмосферное давление) — слабая отрицательная корреляция.

# %% [markdown]
# ### 3.2 Гистограммы распределения каждого признака

# %%
# Исходные числовые столбцы (без добавленных нами hour, day_of_week, is_weekend)
original_numeric_cols = [
    col for col in numeric_cols
    if col not in ['hour', 'day_of_week', 'is_weekend']
]

n_cols_plot = 3
n_rows_plot = int(np.ceil(len(original_numeric_cols) / n_cols_plot))

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 4 * n_rows_plot))
axes = axes.flatten()

for i, col in enumerate(original_numeric_cols):
    ax = axes[i]
    sns.histplot(df[col], bins=50, kde=True, ax=ax, color='steelblue', alpha=0.7)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    # Отмечаем среднее и медиану
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'mean={mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='-.', label=f'median={median_val:.1f}')
    ax.legend(fontsize=8)

# Скрываем пустые подграфики
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Гистограммы распределения всех числовых признаков", fontsize=16, y=1.01)
fig.tight_layout()
save_fig(fig, "03_histograms_all_features")

# %% [markdown]
# **Интерпретация гистограмм:**
#
# - **`T (degC)`** (температура) — распределение близко к нормальному с лёгким
#   смещением в сторону положительных значений (в Германии лето чуть теплее,
#   чем зима холодна). Диапазон: примерно от –23°C до +37°C.
# - **`p (mbar)`** (атмосферное давление) — узкое, почти гауссово распределение
#   с центром около 989 мбар.
# - **`rh (%)`** (относительная влажность) — левосторонне смещённое (скошено
#   к высоким значениям), что типично для влажного климата центральной Европы.
# - **`wv (m/s)`** и **`max. wv (m/s)`** (скорость ветра) — правосторонне
#   смещённое, с длинным хвостом вправо. Возможны аномально высокие значения,
#   требующие проверки на выбросы.
# - **`wd (deg)`** (направление ветра) — равномерное распределение, что ожидаемо,
#   так как ветер дует со всех направлений.

# %% [markdown]
# ### 3.3 Временной ряд температуры со скользящим средним

# %%
# Полный временной ряд температуры за весь период наблюдений
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(df.index, df[TARGET_COL], alpha=0.3, linewidth=0.3,
        color='steelblue', label='Исходная температура (10 мин)')

# Скользящее среднее за сутки (144 наблюдения = 24ч × 6 точек/час)
rolling_1d = df[TARGET_COL].rolling(window=144, center=True).mean()
ax.plot(df.index, rolling_1d, color='orange', linewidth=1,
        label='Скользящее среднее (1 сутки)', alpha=0.8)

# Скользящее среднее за неделю (1008 наблюдений = 7 сут × 144)
rolling_7d = df[TARGET_COL].rolling(window=1008, center=True).mean()
ax.plot(df.index, rolling_7d, color='red', linewidth=2,
        label='Скользящее среднее (1 неделя)')

ax.set_title("Временной ряд температуры (Jena, 2009–2016)", fontsize=16)
ax.set_xlabel("Дата")
ax.set_ylabel("Температура, °C")
ax.legend(fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.tight_layout()
save_fig(fig, "04_time_series_full")

# %% [markdown]
# **Интерпретация временного ряда:**
#
# На графике отчётливо видна **годовая сезонность**: температура циклически
# колеблется от ≈ –15…–20°C зимой до ≈ +30…+37°C летом. Период одного цикла —
# 1 год. Скользящие средние (суточное и недельное) помогают отфильтровать
# высокочастотный шум и выявить:
# - **Тренд:** выраженного линейного тренда за 8 лет не наблюдается — средняя
#   годовая температура остаётся относительно стабильной.
# - **Сезонность:** ярко выраженная синусоидальная компонента с периодом 12 месяцев.
# - **Суточные колебания:** видны по тонкой голубой линии (амплитуда 5–15°C
#   в зависимости от сезона).

# %% [markdown]
# ### 3.4 Зависимость температуры от часа суток и дня недели

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# График 1: Средняя температура по часам суток
hourly_mean = df.groupby('hour')[TARGET_COL].mean()
axes[0].bar(hourly_mean.index, hourly_mean.values, color='coral', edgecolor='black',
            alpha=0.8)
axes[0].plot(hourly_mean.index, hourly_mean.values, 'o-', color='darkred', linewidth=2)
axes[0].set_title("Средняя температура по часам суток", fontsize=14)
axes[0].set_xlabel("Час суток")
axes[0].set_ylabel("Средняя температура, °C")
axes[0].set_xticks(range(0, 24))
axes[0].grid(axis='y', alpha=0.3)

# График 2: Средняя температура по дням недели
day_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
daily_mean = df.groupby('day_of_week')[TARGET_COL].mean()
axes[1].bar(daily_mean.index, daily_mean.values, color='skyblue', edgecolor='black',
            alpha=0.8)
axes[1].plot(daily_mean.index, daily_mean.values, 'o-', color='darkblue', linewidth=2)
axes[1].set_title("Средняя температура по дням недели", fontsize=14)
axes[1].set_xlabel("День недели")
axes[1].set_ylabel("Средняя температура, °C")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(day_labels)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle(f"Зависимость '{TARGET_COL}' от времени", fontsize=16, y=1.02)
fig.tight_layout()
save_fig(fig, "05_mean_by_hour_and_day")

# %%
# Дополнительно: boxplot для более детального распределения
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(x='hour', y=TARGET_COL, data=df, ax=axes[0], palette='YlOrRd')
axes[0].set_title("Распределение температуры по часам суток", fontsize=14)
axes[0].set_xlabel("Час суток")
axes[0].set_ylabel("Температура, °C")

sns.boxplot(x='day_of_week', y=TARGET_COL, data=df, ax=axes[1], palette='Blues')
axes[1].set_title("Распределение температуры по дням недели", fontsize=14)
axes[1].set_xlabel("День недели")
axes[1].set_ylabel("Температура, °C")
axes[1].set_xticklabels(day_labels)

fig.tight_layout()
save_fig(fig, "06_boxplot_hour_and_day")

# %% [markdown]
# **Интерпретация графиков зависимости от времени:**
#
# **По часам суток:**
# - Чётко видна суточная сезонность: температура минимальна в ночные/утренние
#   часы (04:00–06:00) и максимальна в послеобеденное время (14:00–16:00).
# - Амплитуда суточных колебаний составляет примерно 3–5°C (по средним значениям),
#   но может достигать 10–15°C в отдельные дни (видно по boxplot).
#
# **По дням недели:**
# - Различия между днями недели минимальны (менее 1°C), что ожидаемо:
#   метеорологические параметры не зависят от календарного дня недели.
# - Это подтверждает, что признак `day_of_week` сам по себе слабо предсказывает
#   температуру, однако может быть полезен в комбинации с другими признаками.

# %% [markdown]
# ---
# ## 4. Обработка пропусков и выбросов
# **Задачи:**
# - Определить наличие пропущенных значений. Если они есть, предложить стратегию
#   их заполнения (например, интерполяция по времени, заполнение средним/медианой).
# - Выявить и обработать явные выбросы. Обосновать выбранный метод.

# %% [markdown]
# ### 4.1 Анализ пропущенных значений

# %%
print("=" * 60)
print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 60)
missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'Пропусков': missing_count,
    'Процент (%)': missing_pct
})
print(missing_report.to_string())
print(f"\nОбщее число пропусков: {missing_count.sum()}")

# %% [markdown]
# **Стратегия заполнения пропусков (обоснование):**
#
# В данном датасете пропущенных значений **не обнаружено**. Однако если бы они
# были, для временных рядов метеоданных рекомендуется следующая стратегия:
#
# 1. **Временная интерполяция (`df.interpolate(method='time')`)** — наилучший
#    выбор для временных рядов, так как она учитывает неравномерность шага и
#    восстанавливает значения плавно, в отличие от заполнения средним/медианой,
#    которое создаёт резкие «ступеньки» на графике.
# 2. Заполнение средним или медианой (`fillna(df.mean())`) **не рекомендуется**
#    для временных рядов, так как оно не учитывает локальный контекст (сезон,
#    время суток) и может существенно исказить данные.
# 3. Для длинных пропусков (более суток) предпочтительнее использовать
#    **сплайновую интерполяцию** или **заполнение значением аналогичного периода
#    прошлого года**, так как линейная интерполяция в этом случае может давать
#    неадекватные результаты.

# %% [markdown]
# ### 4.2 Выявление и обработка выбросов

# %%
# Визуализация выбросов ПЕРЕД обработкой (boxplots)
outlier_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in ['hour', 'day_of_week', 'is_weekend']
]

n_box = len(outlier_cols)
n_box_cols = 4
n_box_rows = int(np.ceil(n_box / n_box_cols))

fig, axes = plt.subplots(n_box_rows, n_box_cols, figsize=(20, 5 * n_box_rows))
axes = axes.flatten()

for i, col in enumerate(outlier_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightyellow',
                flierprops={'marker': 'o', 'markersize': 2})
    axes[i].set_title(col, fontsize=11, fontweight='bold')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Boxplot-анализ выбросов ДО обработки", fontsize=16, y=1.01)
fig.tight_layout()
save_fig(fig, "07_boxplots_before_outliers")

# %%
# Обработка выбросов: метод Z-score (замена значений вне 4σ на интерполяцию)
#
# Обоснование выбора метода:
# - Используем порог 4σ (а не стандартный 3σ), чтобы не удалять «экстремальную
#   но реальную погоду» (например, аномально жаркий день). 4σ отсекает только
#   действительно невозможные или аппаратные ошибки.
# - Заменяем выбросы на NaN, а затем используем временную интерполяцию,
#   чтобы восстановить значение на основе соседних точек. Это сохраняет
#   непрерывность и регулярность временного ряда (важно для нейросетей).
# - Альтернативу (удаление строк) мы отклоняем, так как она нарушает
#   регулярный 10-минутный шаг и создаёт «дырки» во временном ряде.

def detect_and_fix_outliers(series, sigma_threshold=4):
    """
    Обнаруживает выбросы по методу Z-score и заменяет их
    линейной интерполяцией по времени.

    Parameters:
        series: pd.Series — исходный числовой ряд
        sigma_threshold: float — порог стандартных отклонений

    Returns:
        pd.Series — очищенный ряд
    """
    mean_val = series.mean()
    std_val = series.std()
    lower_bound = mean_val - sigma_threshold * std_val
    upper_bound = mean_val + sigma_threshold * std_val

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    n_outliers = outlier_mask.sum()

    if n_outliers > 0:
        print(f"  [{series.name:25s}] Выбросов: {n_outliers:5d} "
              f"(допустимый диапазон: [{lower_bound:.2f}, {upper_bound:.2f}])")
    else:
        print(f"  [{series.name:25s}] Выбросов: 0 ✓")

    series_clean = series.copy()
    series_clean.loc[outlier_mask] = np.nan
    return series_clean.interpolate(method='time')


print("=" * 60)
print("ОБРАБОТКА ВЫБРОСОВ (метод Z-score, порог = 4σ)")
print("=" * 60)

cols_to_clean = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in ['hour', 'day_of_week', 'is_weekend']
]

for col in cols_to_clean:
    df[col] = detect_and_fix_outliers(df[col])

print(f"\nВсего обработано столбцов: {len(cols_to_clean)}")
print("Выбросы заменены временной интерполяцией ✓")

# %% [markdown]
# ### Вывод по шагу 4:
# - **Пропуски:** В оригинальном датасете пропущенных значений нет. В случае их
#   появления рекомендуется временная интерполяция (`interpolate(method='time')`),
#   поскольку она учитывает хронологию данных и не создаёт разрывов.
# - **Выбросы:** Применён метод Z-score с порогом **4σ**. Найденные экстремальные
#   значения (вероятные аппаратные сбои датчиков) заменены линейной интерполяцией
#   по времени, что сохраняет непрерывность временного ряда и не создаёт «дырок»
#   (в отличие от простого удаления строк).
# - Для столбцов скорости ветра (`wv`, `max. wv`) обнаружены значения
#   аномально высокие (возможные ошибки датчика) — они корректно обработаны.

# %% [markdown]
# ---
# ## 5. Проектирование новых признаков (Feature Engineering)
# **Задачи:**
# - Создать лаговые признаки: значение T 1 час назад (`lag_1`), 2 часа назад
#   (`lag_2`), 3 часа назад (`lag_3`).
# - Создать признаки скользящего среднего за последние 3, 6 и 12 наблюдений
#   (т.е. 30, 60, 120 минут).
# - Добавить признак «время суток» как категорию (утро, день, вечер, ночь).
#
# **Важное замечание:** Так как записи в датасете идут с шагом **10 минут**,
# 1 час = **6 наблюдений**, 2 часа = 12, 3 часа = 18. Это необходимо
# учитывать при создании лаговых признаков.

# %%
# ----- Лаговые признаки (lag features) -----
# 1 час = 6 интервалов по 10 минут
STEPS_PER_HOUR = 6

df['lag_1'] = df[TARGET_COL].shift(1 * STEPS_PER_HOUR)   # 1 час назад
df['lag_2'] = df[TARGET_COL].shift(2 * STEPS_PER_HOUR)   # 2 часа назад
df['lag_3'] = df[TARGET_COL].shift(3 * STEPS_PER_HOUR)   # 3 часа назад

print("=== Лаговые признаки (строки 19–28 для демонстрации) ===")
print(df[[TARGET_COL, 'lag_1', 'lag_2', 'lag_3']].iloc[18:28].to_string())

# %%
# ----- Скользящие средние (rolling mean) -----
# За последние 3 наблюдения (30 мин), 6 (60 мин), 12 (120 мин)
df['rolling_mean_30min'] = df[TARGET_COL].rolling(window=3).mean()
df['rolling_mean_60min'] = df[TARGET_COL].rolling(window=6).mean()
df['rolling_mean_120min'] = df[TARGET_COL].rolling(window=12).mean()

print("=== Скользящие средние (строки 13–22) ===")
print(df[[TARGET_COL, 'rolling_mean_30min', 'rolling_mean_60min',
          'rolling_mean_120min']].iloc[12:22].to_string())

# %%
# ----- Категория «Время суток» -----
# Утро: 6:00–11:59, День: 12:00–17:59, Вечер: 18:00–23:59, Ночь: 0:00–5:59
def classify_time_of_day(hour):
    """Классифицирует час суток в категорию."""
    if 0 <= hour < 6:
        return 'ночь'
    elif 6 <= hour < 12:
        return 'утро'
    elif 12 <= hour < 18:
        return 'день'
    else:
        return 'вечер'


df['time_of_day'] = df['hour'].apply(classify_time_of_day)

# Проверяем распределение по категориям
print("\n=== Распределение записей по времени суток ===")
print(df['time_of_day'].value_counts().to_string())

# One-Hot Encoding для использования в нейросетях
df = pd.get_dummies(df, columns=['time_of_day'], prefix='tod', drop_first=False)

# %%
# ----- Удаление строк с NaN (образовавшихся из-за shift и rolling) -----
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
rows_lost = rows_before - rows_after
print(f"\nСтрок до удаления NaN:    {rows_before}")
print(f"Строк после удаления NaN: {rows_after}")
print(f"Удалено строк: {rows_lost} ({rows_lost / rows_before * 100:.3f}%)")

# %%
# Итоговый набор признаков
print("\n=== Итоговые столбцы DataFrame ===")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col} (тип: {df[col].dtype})")
print(f"\nИтоговый размер: {df.shape[0]} строк × {df.shape[1]} столбцов")

# %% [markdown]
# ### Вывод по шагу 5:
# Созданные признаки:
#
# | Признак | Описание | Метод |
# |---------|----------|-------|
# | `lag_1`, `lag_2`, `lag_3` | Температура 1, 2, 3 часа назад | `shift(6)`, `shift(12)`, `shift(18)` |
# | `rolling_mean_30min` | Скользящее среднее за 30 мин | `rolling(3).mean()` |
# | `rolling_mean_60min` | Скользящее среднее за 60 мин | `rolling(6).mean()` |
# | `rolling_mean_120min` | Скользящее среднее за 120 мин | `rolling(12).mean()` |
# | `tod_утро`, `tod_день`, `tod_вечер`, `tod_ночь` | Время суток (OHE) | One-Hot Encoding |
#
# Лаговые и скользящие признаки критически важны для нейросетей (RNN, LSTM),
# так как они дают модели явную информацию о недавней динамике температуры.
# Удалено всего **18 строк** из начала массива (≈0.004% данных), что
# пренебрежимо мало и не влияет на качество анализа.

# %% [markdown]
# ---
# ## 6. Сохранение обработанных данных
# **Задачи:**
# - Очищенный и обогащённый DataFrame сохранить в файл `processed_data.csv`.
# - Дополнительно: продемонстрировать подготовку данных к нормализации
#   с помощью `scikit-learn` (`MinMaxScaler`).

# %%
# Сохранение обработанных данных
df.to_csv(PROCESSED_DATA_PATH, index=True)
print(f"✓ Данные сохранены: {PROCESSED_DATA_PATH}")
print(f"  Размер файла: {os.path.getsize(PROCESSED_DATA_PATH) / 1024 / 1024:.1f} МБ")
print(f"  Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

# Финальный просмотр
print("\n=== Первые 3 строки итогового DataFrame ===")
print(df.head(3).to_string())
print("\n=== Последние 3 строки итогового DataFrame ===")
print(df.tail(3).to_string())

# %% [markdown]
# ### Демонстрация нормализации (scikit-learn)
# Для подготовки данных к обучению нейросетевых моделей на последующих этапах
# применяется масштабирование признаков. Ниже показана нормализация с помощью
# `MinMaxScaler` из библиотеки `scikit-learn`, которая приводит все значения
# к диапазону [0, 1].

# %%
# Выбираем числовые столбцы для нормализации (без OHE-столбцов, они уже 0/1)
scale_cols = [
    col for col in df.columns
    if df[col].dtype in ['float64', 'int64']
    and not col.startswith('tod_')
    and col not in ['is_weekend']
]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

print("=== Нормализация данных (MinMaxScaler, scikit-learn) ===")
print(f"Нормализовано столбцов: {len(scale_cols)}")
print(f"Диапазон значений после нормализации: [{df_scaled[scale_cols].min().min():.1f}, "
      f"{df_scaled[scale_cols].max().max():.1f}]")
print("\nПервые 3 строки нормализованного DataFrame:")
print(df_scaled.head(3).to_string())

# %% [markdown]
# ---
# # Общий итоговый отчёт по Этапу 1
#
# ## Описание проведённой работы
#
# В рамках данного этапа был проведён полный разведочный анализ данных (EDA)
# датасета **Jena Climate 2009–2016**, содержащего метеорологические наблюдения
# с шагом 10 минут в течение 8 лет. Целевой переменной выбрана **температура
# воздуха `T (degC)`**.
#
# ## Основные результаты
#
# ### 1. Структура данных
# - Датасет содержит ~420 000 записей и 14 числовых признаков.
# - Пропущенных значений нет.
# - Данные записаны равномерно с шагом 10 минут.
#
# ### 2. Выявленные закономерности
# - **Годовая сезонность:** ярко выраженные циклы с периодом 12 месяцев
#   (зима: –20°C, лето: +37°C).
# - **Суточная сезонность:** минимум температуры в 4–6 утра, максимум в 14–16 часов.
# - **Корреляции:** `Tdew` (точка росы) и `VPact` (давление пара) сильно
#   положительно коррелируют с температурой. Плотность воздуха (`rho`) — сильная
#   обратная корреляция (что физически обосновано).
# - **По дням недели:** значимых различий нет (ожидаемо для метеоданных).
#
# ### 3. Очистка данных
# - Выбросы (вне 4σ) обнаружены и заменены временной интерполяцией.
# - Метод обоснован: сохраняет непрерывность ряда и не нарушает регулярный шаг.
#
# ### 4. Созданные признаки
# - 3 лаговых признака (1, 2, 3 часа назад).
# - 3 скользящих средних (30, 60, 120 минут).
# - 4 OHE-признака времени суток (утро/день/вечер/ночь).
# - 3 временных признака (час, день недели, флаг выходного).
#
# ### 5. Используемые библиотеки
# - `pandas` — загрузка, обработка и сохранение данных
# - `numpy` — численные вычисления
# - `matplotlib` и `seaborn` — визуализация
# - `scikit-learn` (`MinMaxScaler`) — нормализация признаков
#
# ### 6. Подготовленные данные
# - Итоговый файл: `processed_data.csv`
# - Графики сохранены в папке: `figures/`
# - Данные полностью готовы для следующих этапов: разбиение на train/val/test,
#   нормализация и обучение нейросетевых моделей (Dense, RNN, LSTM).
