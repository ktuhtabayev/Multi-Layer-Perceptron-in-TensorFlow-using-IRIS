
# Import the necessary libraries.
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# Load the data and split it into training and testing sets.
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the input and output layers.
inputs = tf.keras.layers.Input(shape=(4,))
outputs = tf.keras.layers.Dense(3, activation="softmax")(inputs)



# Define the hidden layers
hidden1 = tf.keras.layers.Dense(10, activation="relu")(inputs)
hidden2 = tf.keras.layers.Dense(10, activation="relu")(hidden1)
outputs = tf.keras.layers.Dense(3, activation="softmax")(hidden2)



# Define the model and compile it
model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)



# Train the model on the training set
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))



# Evaluate the performance of the model on the testing set.
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
