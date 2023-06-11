https://github.com/ZulaikhaZamri/Capstone3zulaikha.git
The source of the data: https://data.mendeley.com/datasets/5y9wdsg2zt/2

Project Title:"Concrete Crack Classification: Enhancing Building Safety with AI"

Objectives: To create an image classification model that effectively categorizes concrete images as either cracked or crack-free

Problem Statement: There are several challenges that may be encountered when developing an image classification model for concrete crack detection. As an example:
1)Preparing the data for training, including resizing, normalizing, and augmenting the images, can be complex. Finding the right balance of data augmentation techniques to improve the model's generalization while avoiding overfitting is crucial
2) Selecting the most suitable model architecture and hyperparameters for the concrete crack classification task can be challenging
3)Integrating the trained model into existing systems or workflows, ensuring its compatibility, scalability, and efficiency, can be challenging.

Here are some features that could be implemented in a concrete crack classification system:
1)Model monitoring and retraining
2)Model explainability
3)Mobile or edge deployment

Here's an alternative method using transfer learning :
from google.colab import drive
drive.mount('/content/drive')

#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import matplotlib.pyplot as plt
import os,datetime

!unzip /content/drive/MyDrive/Capstone3.zip

import os
import sys
import subprocess

# Specify the path to the RAR file
rar_path = "/content/Concrete Crack Images for Classification.rar"

# Specify the directory to extract the contents to
extract_dir = "/content/Concrete Crack Images for Classification_images"

# Create the directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Extract the contents of the RAR file using the unrar command-line tool
command = ["unrar", "x", rar_path, extract_dir]
subprocess.call(command)

print("Extraction completed.")

#2. Data loading
file_path = r"/content/Concrete Crack Images for Classification_images"

BATCH_SIZE = 32
IMG_SIZE = (224,224)

data = keras.utils.image_dataset_from_directory(r"/content/Concrete Crack Images for Classification_images", batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data) * 0.1)

train_dataset = data.take(train_size)
val_dataset = data.skip(train_size).take(val_size)
test_dataset = data.skip(train_size+val_size).take(test_size)

class_names = data.class_names
print(class_names)

#4. Splitting the val_dataset into validation and test datasets
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#5. Convert the datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#6. Create a Keras model for data augmentation
data_augmentation = keras.Sequential(name='data_augmentation')
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#7. Test out the data augmentation model
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        # Apply augmentation
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
plt.show()

#8. Create a layer to perform the pixel standardization
preprocess_input = applications.mobilenet_v2.preprocess_input

#9. Start to apply transfer learning
#(A) Get the pretrained model (only the feature extractor)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#(B) Set the pretrained feature extractor as non-trainable (freezing)
base_model.trainable = False
base_model.summary()

#(C) Build our own classifier
# Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()

# Create output layer
output_layer = layers.Dense(len(class_names),activation='softmax')

#(D) Create the final model that contains the entire pipeline
#i. Input layer
inputs = keras.Input(shape=IMG_SHAPE)
#ii. Data augmentation model
x = data_augmentation(inputs)
#iii. Pixel standardization layer
x = preprocess_input(x)
#iv. Feature extraction layers
x = base_model(x,training=False)
#v. Global average pooling layer
x = global_avg(x) 
#vi. Output layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#(E) Instantiate the final model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#10. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#11. Evaluate the model before training
loss0, acc0 = model.evaluate(pf_test)
print("----------------------Evaluation before training-----------------------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)

#12. Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorbaord_logs\transfer_learning_demo"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

#13. Model training
EPOCHS = 2
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])

#14. Proceed with the follow-up training
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False #Freezing the first 100 layers
base_model.summary()

#15. Recompile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#16. Continue with the model training
fine_tune_epoch = 2
total_epoch = EPOCHS + fine_tune_epoch
# Follow-up training
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch = history.epoch[-1],callbacks=[tb])

#17. Evaluate the final transfer learning model
test_loss, test_accuracy = model.evaluate(pf_test)
print("--------------Evaluation after training-------------------")
print("Test loss = ",test_loss)
print("Test accuracy = ",test_accuracy)

#18. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)
# Stack the label and prediction in one numpy array for comparison
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))

#19. Save the model
save_path = os.path.join("save_model","concrete_crack_images_model.h5")
model.save(save_path)
