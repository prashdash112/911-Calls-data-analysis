from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

housing=fetch_california_housing()
X_train_full,X_test,y_train_full,y_test=train_test_split(
    housing.data, housing.target,train_size=0.7)
X_train,X_valid,y_train,y_valid=train_test_split(
    X_train_full,y_train_full,train_size=0.7)

scaler1=StandardScaler()
X_train=scaler1.fit_transform(X_train)
X_valid=scaler1.transform(X_valid)
x_test=scaler1.transform(X_test)



model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(30,activation='relu',input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mean_squared_error,optimizer='sgd'
             )
history=model.fit(X_train,y_train,epochs=60,
                  validation_data=(X_valid,y_valid))

mse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)
