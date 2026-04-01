import os
import sys

# --- CUDA Hotfix for TensorFlow Virtual Environments ---
# Force TensorFlow to find the pip-installed NVIDIA GPU libraries on Linux
if 'LD_LIBRARY_PATH_PATCHED' not in os.environ:
    import site
    import glob
    site_packages = site.getsitepackages()[0]
    nvidia_libs = glob.glob(f"{site_packages}/nvidia/*/lib")
    if nvidia_libs:
        os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':' + ':'.join(nvidia_libs)
        os.environ['LD_LIBRARY_PATH_PATCHED'] = '1'
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Подготовка данных для регрессии (LSTM)
# ==========================================
# Загрузка данных
script_dir = os.path.dirname(os.path.abspath(__file__))
# Data is located in assignment_2 folder
data_path = os.path.normpath(os.path.join(script_dir, '../assignment_2/processed_data.csv'))
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)

# Выбор целевой переменной
target_col = 'Appliances' if 'Appliances' in df.columns else 'T (degC)'
drop_cols = ['Date Time'] # Исключаем временные метки

X_raw = df.drop(columns=[target_col] + drop_cols, errors='ignore')
y_raw = df[target_col]

# Оставляем только числовые признаки для нейросети
X_raw = X_raw.select_dtypes(include=[np.number]).astype(np.float32)
y_raw = y_raw.astype(np.float32)

# Создаем папку для графиков
fig_dir = os.path.join(script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Функция для создания последовательностей временных рядов (Memory Optimized)
def create_sequences(X, y, time_steps=24):
    X_mat = X.values.astype(np.float32)
    y_mat = y.values.astype(np.float32)
    
    n_samples = len(X_mat) - time_steps
    Xs = np.empty((n_samples, time_steps, X_mat.shape[1]), dtype=np.float32)
    ys = np.empty((n_samples,), dtype=np.float32)
    
    for i in range(n_samples):
        Xs[i] = X_mat[i : i + time_steps]
        ys[i] = y_mat[i + time_steps]
        
    return Xs, ys

def run_experiment(window_length):
    print(f"\n========================================================")
    print(f"Начало эксперимента с длиной окна: {window_length} таймстепов")
    print(f"========================================================")
    
    # 1. Создаем последовательности
    X_seq, y_seq = create_sequences(X_raw, y_raw, time_steps=window_length)
    
    # Хронологическое разделение: 70% train, 15% val, 15% test
    n = len(X_seq)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    # 2. Стандартизация (масштабируем 3D тензор)
    # Scaler должен быть обучен на 2D данных
    num_features = X_train.shape[2]
    scaler = StandardScaler()
    
    # Reshape train to 2D for fitting scaler
    X_train_2d = X_train.reshape(-1, num_features)
    scaler.fit(X_train_2d)
    
    # Transform and reshape back to 3D
    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    
    print(f"Train stats: X={X_train_scaled.shape}, y={y_train.shape}")
    
    # 3. Построение модели LSTM
    model = Sequential([
        Input(shape=(window_length, num_features)),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Уменьшаем epochs и увеличиваем batch_size для экономии времени при подборе
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=15, 
        batch_size=256, 
        validation_data=(X_val_scaled, y_val), 
        callbacks=[early_stop], 
        verbose=1
    )
    
    # 4. Оценка качества
    def evaluate_model(mod, X_t, y_t, X_v, y_v, X_te, y_te):
        metrics = {}
        for name, X_data, y_data in zip(['Train', 'Val', 'Test'], [X_t, X_v, X_te], [y_t, y_v, y_te]):
            preds = mod.predict(X_data, batch_size=512, verbose=0).flatten()
            metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(y_data, preds))
            metrics[f'{name}_MAE'] = mean_absolute_error(y_data, preds)
        return metrics
    
    metrics = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Сохраняем только предсказания и результаты для минимизации потребления RAM
    preds = model.predict(X_test_scaled, batch_size=512, verbose=0).flatten()
    
    return model, history, metrics, preds, y_test

# ==========================================
# 2. Анализ влияния длины окна
# ==========================================
windows = [12, 24, 48]
results = {}

for w in windows:
    model, history, metrics, preds, y_test = run_experiment(w)
    results[w] = {
        'model': model if w == 24 else None, # Сохраняем в памяти только одну 'лучшую/среднюю' модель (24)
        'history': history,
        'metrics': metrics,
        'preds': preds,
        'y_test': y_test
    }
    # Очищаем Session и сборка мусора чтобы предотвартить OOM
    import gc
    from tensorflow.keras import backend as K
    K.clear_session()
    gc.collect()

# Собираем метрики в одну таблицу
metrics_df = pd.DataFrame([res['metrics'] for w, res in results.items()], index=[f'LSTM_window_{w}' for w in windows])

# Добавляем результаты MLP (из assignment 2, примерные цифры для сравнения, если они не сохранились, их можно просто вывести для сравнения в отчете)
print("\n=== Сравнение метрик (разные длины окон LSTM) ===")
print(metrics_df)

# Находим лучшее окно (по Test_RMSE)
best_window = metrics_df['Test_RMSE'].idxmin().split('_')[-1]
best_window = int(best_window)
print(f"\nЛучший размер окна по Test RMSE: {best_window}")

best_model = results[best_window]['model']
best_history = results[best_window]['history']
best_preds = results[best_window]['preds']
best_y_test = results[best_window]['y_test']

# ==========================================
# 3. Визуализация для лучшей модели
# ==========================================

# График 1: Факт vs Предсказание (возьмем часть данных)
plt.figure(figsize=(15, 6))
plot_size = 500

plt.plot(best_y_test[:plot_size], label='Факт', color='black', linewidth=1.5)
plt.plot(best_preds[:plot_size], label=f'LSTM pred (window={best_window})', linestyle='dashed', color='red')
plt.title(f'Факт — Предсказание на тестовой выборке (первые {plot_size} значений)')
plt.xlabel('Временной шаг')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_dir, '01_lstm_fact_vs_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()

# График 2: Динамика обучения (Loss)
plt.figure(figsize=(12, 6))
plt.plot(best_history.history['loss'], label='LSTM Обучение')
plt.plot(best_history.history['val_loss'], label='LSTM Валидация')
plt.title(f'Динамика функций потерь (MSE) - Window {best_window}')
plt.xlabel('Эпоха')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(fig_dir, '02_lstm_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# График 3: Сравнение длин окон (RMSE/MAE)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
time_steps = [str(w) for w in windows]
rmse_vals = [results[w]['metrics']['Test_RMSE'] for w in windows]
mae_vals = [results[w]['metrics']['Test_MAE'] for w in windows]

ax[0].bar(time_steps, rmse_vals, color='skyblue')
ax[0].set_title('Test RMSE vs Window Length')
ax[0].set_xlabel('Длина окна (таймстепы)')
ax[0].set_ylabel('RMSE')

ax[1].bar(time_steps, mae_vals, color='salmon')
ax[1].set_title('Test MAE vs Window Length')
ax[1].set_xlabel('Длина окна (таймстепы)')
ax[1].set_ylabel('MAE')

plt.savefig(os.path.join(fig_dir, '03_window_length_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 4. Сохранение лучшей модели
# ==========================================
model_path = os.path.join(script_dir, 'lstm_best.h5')
best_model.save(model_path)
print(f"Лучшая модель (window={best_window}) сохранена: {model_path}")
print("Эксперимент успешно завершён!")
