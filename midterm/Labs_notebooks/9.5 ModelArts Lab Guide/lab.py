import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Загрузка датасета в папку проекта (скачается один раз ~200MB)
PROJECT_DIR = pathlib.Path(__file__).parent
DATASET_DIR = PROJECT_DIR / 'data'
DATASET_DIR.mkdir(exist_ok=True)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(
    'flower_photos.tgz',
    origin=dataset_url,
    extract=True,
    cache_dir=str(DATASET_DIR),
    cache_subdir='',
)
archive_path = pathlib.Path(archive)
candidates = [
    archive_path.parent / 'flower_photos_extracted' / 'flower_photos',
    archive_path.parent / 'flower_photos',
]
data_dir = next((p for p in candidates if p.exists()), None)
if data_dir is None:
    raise FileNotFoundError(f"Не найдена папка flower_photos рядом с {archive_path}")
print("Датасет:", data_dir)

# 2. Параметры
IMG_SIZE = 180
BATCH_SIZE = 32
EPOCHS = 10

# 3. Загрузка данных
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Классы:", class_names)  # daisy, dandelion, roses, sunflowers, tulips

# 3.1 Превью примеров из датасета
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.suptitle('Примеры из датасета')
plt.tight_layout()
samples_path = PROJECT_DIR / 'dataset_samples.png'
plt.savefig(samples_path, dpi=150)
print(f"Примеры сохранены: {samples_path}")
plt.show()

# 4. Оптимизация загрузки
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. Модель (Transfer Learning — быстро и точно)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

# 6. Компиляция
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Обучение
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 8. Графики обучения: точность и потери
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Flower Classification Training')
plt.tight_layout()
output_path = PROJECT_DIR / 'training_curves.png'
plt.savefig(output_path, dpi=150)
print(f"График сохранён: {output_path}")
plt.show()

import numpy as np
from tensorflow.keras.preprocessing import image

# 9. Confusion matrix по всей валидации
all_true = []
all_pred = []
for imgs, lbls in val_ds:
    p = model.predict(imgs, verbose=0)
    all_pred.append(np.argmax(p, axis=1))
    all_true.append(lbls.numpy())
all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

cm = tf.math.confusion_matrix(all_true, all_pred, num_classes=len(class_names)).numpy()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_xlabel('Предсказано')
ax.set_ylabel('Реально')
ax.set_title('Confusion Matrix (валидация)')
threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > threshold else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
cm_path = PROJECT_DIR / 'confusion_matrix.png'
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix сохранена: {cm_path}")
plt.show()

# 10. Проверка модели на примерах из валидации
images_batch, labels_batch = next(iter(val_ds))
preds = model.predict(images_batch)
pred_labels = np.argmax(preds, axis=1)
true_labels = labels_batch.numpy()

n_show = min(9, len(images_batch))
correct_total = int((pred_labels == true_labels).sum())

plt.figure(figsize=(10, 10))
for i in range(n_show):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images_batch[i].numpy().astype("uint8"))
    is_correct = pred_labels[i] == true_labels[i]
    color = 'green' if is_correct else 'red'
    mark = '✓' if is_correct else '✗'
    plt.title(
        f"{mark} pred: {class_names[pred_labels[i]]}\nreal: {class_names[true_labels[i]]}",
        color=color, fontsize=10
    )
    plt.axis("off")
plt.suptitle(
    f'Распознано правильно: {correct_total} из {len(images_batch)} '
    f'({100*correct_total/len(images_batch):.0f}%)'
)
plt.tight_layout()
predictions_path = PROJECT_DIR / 'predictions.png'
plt.savefig(predictions_path, dpi=150)
print(f"Предсказания сохранены: {predictions_path}")
print(f"Правильно: {correct_total}/{len(images_batch)}")
plt.show()

# 11. Предсказание на одном изображении
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.expand_dims(image.img_to_array(img), 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(f"Результат: {class_names[np.argmax(score)]} ({100*np.max(score):.1f}%)")

# predict_flower("путь/к/фото.jpg")