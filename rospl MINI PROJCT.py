#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize,  LabelBinarizer
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


# Plotting 12 images to check dataset
#Now we will observe some of the iamges that are their in our dataset. We will plot 12 images here using the matplotlib library.
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import cv2
import random
plt.figure(figsize=(12,12))
path = "C:/Users/dhira/Downloads/plant/Plant_images/Potato___Early_blight"
for i in range(1,17):
    plt.subplot(4,4,i)
    plt.tight_layout()
    rand_img = imread(path +'/'+ random.choice(sorted(os.listdir(path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10)#height of image


# In[4]:


#Converting Images to array 
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (256,256))  
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[5]:


from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize,  LabelBinarizer
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
dir = "C:/Users/dhira/Downloads/plant/Plant_images"
root_dir = listdir(dir)
image_list, label_list = [], []
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
binary_labels = [0,1,2]
temp = -1

# Reading and converting image to numpy array
#Now we will convert all the images into numpy array.

for directory in root_dir:
  plant_image_list = listdir(f"{dir}/{directory}")
  temp += 1
  for files in plant_image_list:
    image_path = f"{dir}/{directory}/{files}"
    image_list.append(convert_image_to_array(image_path))
    label_list.append(binary_labels[temp])
    


# In[6]:


# Visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()
label_counts.head()

#it is a balanced dataset as you can see


# In[7]:


#Next we will observe the shape of the image.
image_list[0].shape


# In[8]:


#Checking the total number of the images which is the length of the labels list.
label_list = np.array(label_list)
label_list.shape


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10) 


# In[10]:


#Now we will normalize the dataset of our images. As pixel values ranges from 0 to 255 so we will divide each image pixel with 255 to normalize the dataset.
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape( -1, 256,256,3)
x_test = x_test.reshape( -1, 256,256,3)


# In[11]:


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[12]:


from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256,256,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()


# In[13]:


model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001),metrics=['accuracy'])


# In[14]:


#Next we will split the dataset into validation and training data.
# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)


# In[15]:


# Training the model
epochs = 30
batch_size = 128
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data = (x_val, y_val))


# In[16]:


#Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()


# In[17]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# In[18]:


y_pred = model.predict(x_test)


# In[ ]:


from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from tkinter import messagebox

root = Tk()
root.title("MINI PROJECT")

heading_label = Label(root, text="PLANT HEALTH DETECTION", font=("Helvetica", 18, "bold"))
heading_label.pack()

def showimage():
    # Load the image
    img_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="SELECT file",
                                          filetypes=(("JPG File", ".jpg"), ("PNG File", ".png"), ("ALL Files", ".*")))
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Normalize the image
    img_array = img_array / 255.0

    # Add a new axis to the image array to make it compatible with the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Display the image in the GUI
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img


    # Predict the class of the image using the model
    y_pred = model.predict(img_array)

    # Get the class label corresponding to the predicted class index
    all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
    if y_pred.any():  # Check if y_pred is not empty
        predicted_label = all_labels[np.argmax(y_pred)]
        # Show a message box with the predicted class label
        messagebox.showinfo("Predicted:", predicted_label)
        heading_label1 = Label(root, text="PREVENTIONS FOR PLANT DISEASES", font=("Helvetica", 18, "bold"))
        heading_label1.pack()

        label_text1 = "Quarantine and isolate affected plants: Isolate affected plants from healthy plants to prevent the spread of disease."
        label2 = Label(root, text=label_text1)
        label2.pack()

        label_text2 = "Prune and remove infected plant parts: Prune and remove infected plant parts, such as leaves, stems, or fruits, as soon as they are noticed"
        label3 = Label(root, text=label_text2)
        label3.pack()

        label_text3 = "Properly dispose of infected plant debris: Collect and dispose of infected plant debris properly, such as by burning, burying, or removing from the site."
        label4 = Label(root, text=label_text3)
        label4.pack()

        label_text4 = "Disinfect tools and equipment: Clean and disinfect tools, equipment, and containers that come in contact with infected plants to prevent disease transmission."
        label5 = Label(root, text=label_text4)
        label5.pack()

        label_text5 = "Monitor and scout for disease: Regularly monitor and scout for signs of disease in affected plants and nearby plants"
        label6 = Label(root, text=label_text5)
        label6.pack()

    else:
        messagebox.showerror("Error", "Prediction failed. Make sure the model is loaded.")
        
        

        
btn_open = Button(root, text="Predict", command=showimage)
btn_open.pack()
lbl = Label(root)
lbl.pack()

if __name__ == '__main__':
    root.mainloop()        


# In[43]:


# Finding max value from predition list and comaparing original value vs predicted
print("Originally : ",all_labels[np.argmax(y_test[1])])
print("Predicted : ",all_labels[np.argmax(y_pred[1])])


# In[30]:


plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[31]:


from sklearn.metrics import classification_report

# predict 
pred = model.predict(x_test, batch_size = 32)
pred = np.argmax(pred, axis=1)
# label
y_target = np.argmax(y_test, axis=1)
y_target


# In[32]:


print(classification_report(y_target, pred, target_names = all_labels))


# In[ ]:




