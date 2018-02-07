
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
from scipy.misc import imresize

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.core import Dropout, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

# Tunable parameters
MY_KEEP_PROBS = 0.6
MY_BATCH_SIZE = 128
MY_CORRECTION = 0.22
MY_EPOCHS = 20

#array to receive image path from log file
lines = []

##### Following datasets are used just to verify the over-fitting or underfitting of the model #####
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusData/driving_log.csv') as csvfile:  
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4/driving_log.csv') as csvfile:  
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW/driving_log.csv') as csvfile:  
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack/driving_log.csv') as csvfile:  
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1/driving_log.csv') as csvfile:  
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2/driving_log.csv') as csvfile:   

##### Following are the actual data used for training #####
#open only Track-1 data 
with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2_Smooth3/driving_log.csv') as csvfile:    
#open Track-1 + Track-2 combined data 
#with open('C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2_Smooth3_Track2_Windows/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# split train validation data sets        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# getting histogram of steering angle data
hist_measurement = []
for line in train_samples:    
    hist_measurement.append(float(line[3])*180/3.142) #I think the angle measurements are in radians, so converting them to degrees to display
# print a histogram to see which steering angle ranges are most overrepresented
num_bins = 23
avg_samples_per_bin = len(hist_measurement)/num_bins
hist, bins = np.histogram(hist_measurement, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.figure(0)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(hist_measurement), np.max(hist_measurement)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.title('Histogram of steering angle measurements')
plt.ylabel('# Samples')
plt.xlabel('Steering angle')
    
# data Generator - saves memory    
def generator(samples, batch_size = MY_BATCH_SIZE):     
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            lines = samples[offset:offset+batch_size]
            
            correction = MY_CORRECTION 
            images = []
            measurements = []
            for line in lines:
                for k in range(3):
                    source_path = line[k]
                    
                    if len(source_path.split('/')) > 1: #if train images are generated from a Mac OS Simulator
                        filename = source_path.split('/')[-1] 
                    
                    if len(source_path.split('/')) < 2: #if train images are generated from a Windows Simulator
                        filename = source_path.split('\\')[-1]
                        
                    ##### Following datasets are used just to verify the over-fitting or underfitting of the model #####
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusData/IMG/' + filename
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4/IMG/' + filename
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW/IMG/' + filename
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack/IMG/' + filename
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1/IMG/' + filename
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2/IMG/' + filename
                    
                    #open only Track-1 data
                    current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2_Smooth3/IMG/' + filename
                    #open Track-1 + Track-2 combined data
                    #current_path = 'C:/Users/rksiddak/SDC/Project_3/CarND-Behavioral-Cloning-P3-master/Simulator/SamplePlusMyOwnMac4_FWRW_WeaveBack_Smooth1_Smooth2_Smooth3_Track2_Windows/IMG/' + filename
                    
                    image = cv2.imread(current_path)
                    
                    #BGR to RGB conversion
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
                    images.append(image)
                    
                    #center camera
                    if k == 0:
                        measurement = float(line[3])
                        measurements.append(measurement)
                    
                    #left camera
                    if k == 1:
                        measurement = float(line[3]) + correction
                        measurements.append(measurement)
                        
                    #right camera
                    if k == 2:
                        measurement = float(line[3]) - correction
                        measurements.append(measurement)
                
            augmented_images, augmented_measurements = [], []
            
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1)) #flip image
                augmented_measurements.append(measurement*-1.0) #negate measurement for the flipped image
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=MY_BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=MY_BATCH_SIZE)

#Keras model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) #Lambda layer for input data normalization
model.add(Cropping2D(cropping = ((70,25),(0,0)))) #Crop top 70 and bottom 25 pixels

#Advanced NVIDIA architecture. 
model.add(Convolution2D(24,5,5,subsample = (2,2), activation = "relu")) #, W_regularizer=l2(0.001)
model.add(Convolution2D(36,5,5,subsample = (2,2), activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2), activation = "relu"))

model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Flatten())
model.add(Dropout(MY_KEEP_PROBS))

model.add(Dense(100, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam') #optimizer=Adam(lr=1.0e-4) 
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
                                     validation_data = validation_generator, 
                                     nb_val_samples = len(validation_samples), 
                                     nb_epoch=MY_EPOCHS, verbose=1)
model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure(1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

