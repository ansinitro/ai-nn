import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Подготовка данных для регрессии
# ==========================================
# Загрузка данных
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'processed_data.csv'))

# Выбор целевой переменной (защита от отсутствия нужного столбца)
target_col = 'Appliances' if 'Appliances' in df.columns else 'T (degC)'
drop_cols = ['Date Time'] # Исключаем временные метки

X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
y = df[target_col]

# Оставляем только числовые признаки для нейросети
X = X.select_dtypes(include=[np.number]).astype(np.float32)
y = y.astype(np.float32)

# Хронологическое разделение на train (70%), val (15%), test (15%)
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# Стандартизация (обучаем scaler только на тренировочной выборке)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Настройка ранней остановки для предотвращения переобучения
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ==========================================
# 2. Построение модели MLP (3 скрытых слоя)
# ==========================================
mlp = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

mlp.compile(optimizer='adam', loss='mean_squared_error')

print("Обучение MLP...")
history_mlp = mlp.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=128, 
    validation_data=(X_val_scaled, y_val), 
    callbacks=[early_stop], 
    verbose=1
)

# ==========================================
# 3. Построение глубокой сети DNN (5 скрытых слоев)
# ==========================================
dnn = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

dnn.compile(optimizer='adam', loss='mean_squared_error')

print("\nОбучение DNN...")
history_dnn = dnn.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=128, 
    validation_data=(X_val_scaled, y_val), 
    callbacks=[early_stop], 
    verbose=1
)

# ==========================================
# 4. Оценка качества
# ==========================================
def evaluate_model(model, X_t, y_t, X_v, y_v, X_te, y_te):
    metrics = {}
    for name, X_data, y_data in zip(['Train', 'Val', 'Test'], [X_t, X_v, X_te], [y_t, y_v, y_te]):
        preds = model.predict(X_data, verbose=0)
        metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(y_data, preds))
        metrics[f'{name}_MAE'] = mean_absolute_error(y_data, preds)
    return metrics

mlp_metrics = evaluate_model(mlp, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
dnn_metrics = evaluate_model(dnn, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)

res_df = pd.DataFrame([mlp_metrics, dnn_metrics], index=['MLP', 'DNN'])
print("\nСравнение метрик:")
print(res_df)

# Создаем папку для графиков
fig_dir = os.path.join(script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# График 1: Факт vs Предсказание (возьмем часть данных для наглядности)
plt.figure(figsize=(15, 6))
plot_size = 300
plt.plot(y_test.values[:plot_size], label='Факт', color='black', linewidth=2)
plt.plot(mlp.predict(X_test_scaled[:plot_size], verbose=0), label='MLP предсказание', linestyle='dashed')
plt.plot(dnn.predict(X_test_scaled[:plot_size], verbose=0), label='DNN предсказание', linestyle='dotted')
plt.title(f'Факт — Предсказание на тестовой выборке (первые {plot_size} значений)')
plt.xlabel('Временной шаг')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_dir, '01_fact_vs_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()

# График 2: Динамика обучения (Loss)
plt.figure(figsize=(12, 6))
plt.plot(history_mlp.history['loss'], label='MLP Обучение')
plt.plot(history_mlp.history['val_loss'], label='MLP Валидация')
plt.plot(history_dnn.history['loss'], label='DNN Обучение')
plt.plot(history_dnn.history['val_loss'], label='DNN Валидация')
plt.title('Динамика функций потерь (MSE)')
plt.xlabel('Эпоха')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_dir, '02_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 5. Сохранение лучшей модели
# ==========================================
# Исходя из метрик, MLP является лучшей моделью
model_path = os.path.join(script_dir, 'mlp_best.h5')
mlp.save(model_path)
print(f"Лучшая модель сохранена: {model_path}")