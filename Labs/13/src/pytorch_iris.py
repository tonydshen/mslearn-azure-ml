# pytorch_iris.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# get parameters
parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, default='./iris-data/iris.csv', help="Path to training data")
parser.add_argument("--reg_rate", type=float, default=0.01)
parser.add_argument("--model_output", type=str, default='./model/iris.save', help="Path of output model")
parser.add_argument("--test_size", type=float, default=0.30, help="test size")
parser.add_argument("--random_state", type=int, default=0, help="random state")
parser.add_argument("--n_epoch", type=int, default=50, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
args = parser.parse_args()
print(args)

training_data = args.training_data
reg_rate = args.reg_rate
model_output = args.model_output
test_size = args.test_size
random_state = args.random_state
n_epoch = args.n_epoch
learning_rate = args.learning_rate

# Load Iris dataset
df = pd.read_csv(training_data)  # Replace with your actual file
X = df.drop("species", axis=1).values
y = LabelEncoder().fit_transform(df["species"])
# Assume the last column is the target
# X = df.iloc[:, :-1]  # Features
# y = df.iloc[:, -1]   # Target

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Define model
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_rate)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
# torch.save(model.state_dict(), "iris_model.pt")
torch.save(model.state_dict(), model_output)

# Evaluate
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)
    acc = (preds == y_test).float().mean()
    print(f"Test Accuracy: {acc:.4f}")
