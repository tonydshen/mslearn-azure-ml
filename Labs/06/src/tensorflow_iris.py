# tensorflow_iris.py
import tensorflow as tf
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# get parameters
parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, default='./iris-data/iris.csv', help="Path to training data")
parser.add_argument("--reg_rate", type=float, default=0.01)
parser.add_argument("--model_output", type=str, default='./model/iris_tf.keras', help="Path of output model, file extension .keras instead of .h5 required")
parser.add_argument("--test_size", type=float, default=0.30, help="test size")
parser.add_argument("--random_state", type=int, default=0, help="random state")
parser.add_argument("--n_epoch", type=int, default=50, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")

args = parser.parse_args()
print(args)

training_data = args.training_data
reg_rate = args.reg_rate
model_output = args.model_output
test_size = args.test_size
random_state = args.random_state
n_epoch = args.n_epoch
learning_rate = args.learning_rate
batch_size = args.batch_size

# Load Iris dataset
df = pd.read_csv(training_data)  # Replace with your actual file
X = df.drop("species", axis=1).values
y = LabelEncoder().fit_transform(df["species"])

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(batch_size, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=n_epoch, batch_size=batch_size, verbose=1)

# Save model
model.save(model_output) # Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
