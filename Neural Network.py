import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

#train_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\train.csv")
test_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\test.csv")

#train_DF = pd.DataFrame(train_csv)
test_DF = pd.DataFrame(test_csv)


#TrainingDataTarget = []
#TrainingDataValue = []
TestData = []

#for i in tqdm(range(len(train_DF))):
#    if train_DF.loc[i][0] == 0:
#        target = np.array([0])
#    elif train_DF.loc[i][0] == 1:
#        target = np.array([1])
#    elif train_DF.loc[i][0] == 2:
#        target = np.array([2])
#    elif train_DF.loc[i][0] == 3:
#        target = np.array([3])
#    elif train_DF.loc[i][0] == 4:
#        target = np.array([4])
#    elif train_DF.loc[i][0] == 5:
#        target = np.array([5])
#    elif train_DF.loc[i][0] == 6:
#        target = np.array([6])
#    elif train_DF.loc[i][0] == 7:
#        target = np.array([7])
#    elif train_DF.loc[i][0] == 8:
#        target = np.array([8])
#    elif train_DF.loc[i][0] == 9:
#        target = np.array([9])
#    TrainingDataValue.append(np.array(train_DF.loc[i][1:]).reshape(28, 28)/255.0)
#    TrainingDataTarget.append(target)

#for i in tqdm(range(len(test_DF))):
#    TestData.append(np.array(test_DF.loc[i]).reshape(28, 28)/255.0)

#np.save("TrainingDataTarget", TrainingDataTarget)
#np.save("TrainingDataValue", TrainingDataValue)
#np.save("TestData", TestData)

#TrainingDataLabel = np.load("TrainingDataTarget.npy")
#TrainingDataImages = np.load("TrainingDataValue.npy")

TestData = np.load("TestData.npy")

##print(TrainingDataLabel[6])
##print(TrainingDataImages[6])
##plt.imshow(TrainingDataImages[6])
##plt.show()




#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(128, activation='relu'),
#    keras.layers.Dense(10, activation='softmax')
#])

#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.fit(TrainingDataImages, TrainingDataLabel, epochs=10)

#model.save('MNIST_DATA_using TensorFlow.h5')


new_model = keras.models.load_model('MNIST_DATA_using TensorFlow.h5')

predictions = new_model.predict(TestData)
print("\n\n")

for _ in range(10):
    
    index = np.random.choice(len(TestData))

    print("The Number is {}\n".format(np.argmax(predictions[index])))
    plt.imshow(TestData[index])
    plt.show()
