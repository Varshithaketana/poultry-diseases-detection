from google.colab import drive
drive.mount('/content/drive')
# Load metadata
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/final_metadata_with_paths.csv")
import os

# Path to image folder
image_folder = '/content/drive/MyDrive/train'

# Recursively collect all image file paths
all_images = []
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(root, file)
            all_images.append(full_path)

print(f"🔍 Found {len(all_images)} image files.")
df[['label', 'filename', 'image_path']].head()
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D,
    Concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#  STEP 3: PREPROCESSING
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
df['full_path'] = df['filename'].apply(lambda x: os.path.join("/content/drive/MyDrive/train", x))
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)
meta_columns = ['symptom_lethargy', 'symptom_diarrhea', 'egg_production_rate', 'temperature', 'humidity']
scaler = StandardScaler()
df[meta_columns] = scaler.fit_transform(df[meta_columns])
#train and test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_enc'], random_state=42)
#  STEP 4: MEMORY-SAFE GENERATOR
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class MultiInputGenerator(Sequence):
    def __init__(self, dataframe, meta_columns, num_classes, batch_size=32, img_size=(224, 224), shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = dataframe.reset_index(drop=True)
        self.meta_columns = meta_columns
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_img = np.array([
            img_to_array(load_img(path, target_size=self.img_size)) / 255.0
            for path in batch['full_path']
        ], dtype=np.float32)

        X_meta = batch[self.meta_columns].values.astype(np.float32)
        y = to_categorical(batch['label_enc'], num_classes=self.num_classes)
        return (X_img, X_meta), y
# STEP 5: MODEL DEFINITION (IMAGE + METADATA FUSION)
img_input = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, input_tensor=img_input, weights='imagenet')
base_model.trainable = False
x_img = GlobalAveragePooling2D()(base_model.output)
#meta branch
meta_input = Input(shape=(len(meta_columns),))
x_meta = Dense(64, activation='relu')(meta_input)
x_meta = BatchNormalization()(x_meta)
#fusion
combined = Concatenate()([x_img, x_meta])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=[img_input, meta_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# STEP 6: TRAIN MODEL SAFELY
train_gen = MultiInputGenerator(train_df, meta_columns, num_classes, batch_size=16)
val_gen = MultiInputGenerator(val_df, meta_columns, num_classes, batch_size=16, shuffle=False)
#train_gen = MultiInputGenerator(train_df, batch_size=16)
#val_gen = MultiInputGenerator(val_df, batch_size=16, shuffle=False)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"✅ Validation Accuracy: {acc:.4f}")
#step6-save model
# STEP 7: SAVE MODEL
model.save("/content/multimodal_poultry_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("/content/multimodal_poultry_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\n✅ Model saved as .h5 and .tflite")




