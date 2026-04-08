import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ================= LOAD DATA =================
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

# ================= SPLIT =================
X_train = train.drop("label", axis=1).values
y_train = train["label"].values

X_test = test.drop("label", axis=1).values
y_test = test["label"].values

# ================= RESHAPE =================
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# ================= 🔥 FIX LABEL ISSUE =================
# Dataset has missing labels → remap to 0–23

unique_labels = sorted(np.unique(y_train))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

y_train = np.array([label_map[y] for y in y_train])
y_test = np.array([label_map[y] for y in y_test])

# ================= ONE HOT =================
y_train = to_categorical(y_train, num_classes=24)
y_test = to_categorical(y_test, num_classes=24)

# ================= MODEL =================
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(24, activation="softmax"))  # 24 classes

# ================= COMPILE =================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= TRAIN =================
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# ================= SAVE =================
model.save("cnn_model.h5")

print("✅ Model trained and saved successfully!")