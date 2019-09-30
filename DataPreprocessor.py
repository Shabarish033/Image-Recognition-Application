# -*- coding: utf-8 -*-
#Directory Organiser
import os
import shutil
#get Folder names
import tkinter as tk
from tkinter import filedialog
from tkinter import *
#global path1, path2, TrainingSetPath, TestSetPath, loaded_model
import numpy as np
from keras.models import Sequential # Initialize the Neural Network
from keras.layers import Convolution2D #Create Convolution layer
from keras.layers import MaxPooling2D #Create max pooling layer
from keras.layers import Flatten #convert matrix to vectors
from keras.layers import Dense #Connect the layers
from keras.preprocessing.image import ImageDataGenerator #For image Augmentation
from keras.preprocessing import image
from keras.models import model_from_json #To save the model 

def pathsplit(Path):#This Function splits the path string to makes the path in the window label visible 
    PathSplit = Path.split('/')
    PathSplit = PathSplit[0] + '/' + PathSplit[1] + '/' + PathSplit[2] + '/' + '...' + '/' + PathSplit[-2] +'/'+ PathSplit[-1]
    return PathSplit
def filesearch():
    global path1
    window2 = tk.Tk()
    window2.filename = filedialog.askdirectory()
    path1 = window2.filename
    os.environ['path1'] = path1
    SplitPath = pathsplit(path1)
    u.set(SplitPath)
    window2.destroy()
def filesearch2():
    global path2
    window2 = tk.Tk()
    window2.filename = filedialog.askdirectory()
    path2 = window2.filename
    SplitPath = pathsplit(path2)
    v.set(SplitPath)
    window2.destroy()
def OKAY():
    global TrainingSetPath, TestSetPath
    directory_list = list()
    for root, dirs, files in os.walk(path2, topdown=False):
        for name in dirs:
            directory_list.append(os.path.join(root, name))
    ListAllImages = [[] for i in range(len(dirs))]
    TrainingSetPath = os.mkdir(os.path.join(path1, 'TrainingSet'))
    TrainingSetPath = str(TrainingSetPath)
    os.environ['TrainingSetPath'] = TrainingSetPath
    TestSetPath = os.mkdir(os.path.join(path1, 'TestSet'))
    TestSetPath = str(TestSetPath)
    os.environ['TestSetPath'] = TestSetPath
    
    for directory in range(len(dirs)):
        os.mkdir(os.path.join(path1, 'TrainingSet', str(dirs[directory])))
        os.mkdir(os.path.join(path1, 'TestSet', str(dirs[directory])))
        ListAllImages[directory] = os.listdir(os.path.join(root, str(dirs[directory])))
        print("Folder " + str(dirs[directory]) + " has " + str(int(len(ListAllImages[directory]))) + " Images.")
    
    for directory in range(len(dirs)):
        TestSet = []
        for image in range(0, int(len(ListAllImages[directory])/5)):
            TestSet.append(ListAllImages[directory][image])
            shutil.move(os.path.join(directory_list[directory], TestSet[image]), os.path.join(path1, 'TestSet', str(dirs[directory])))
    
    for directory in range(len(dirs)):
        files = os.listdir(directory_list[directory])
        for file in files:
            shutil.move(os.path.join(directory_list[directory], file), os.path.join(path1, 'TrainingSet', str(dirs[directory])))
    window.destroy()

#User Interface Development
window = tk.Tk()
window.geometry('850x330')
logo = tk.PhotoImage(file="UI/Logo_True.png")
Folder = tk.PhotoImage(file="UI/images.png")
OK = tk.PhotoImage(file="UI/Load.png")
label = tk.Label(window, image= logo).place(x=5, y=10)
label2 = tk.Label(window, text = 'Please provide path to Working Directory',font = "Helvetica 13 bold").place(x=15, y=150)
u = StringVar() #Creating a dummy variable to change text in label
label2_1 = tk.Label(window,anchor='w',height = 1, font='Helvetica 10',textvariable=u,relief='groove', width=45).place(x = 370, y = 150)
u.set('Please Select the working directory...')
button2 = tk.Button(window, image=Folder, command = filesearch).place(x=750, y=140)
Label3 = tk.Label(window, text = 'Please provide path to image files',font = "Helvetica 13 bold").place(x=15, y=200)
v = StringVar() #Creating a dummy variable to change text in label
label3_1 = tk.Label(window,anchor='w',height = 1, font='Helvetica 10',textvariable=v,relief='groove', width=45).place(x = 370, y = 200)
v.set('Please Select the working directory...')
button3 = tk.Button(window, image=Folder, command = filesearch2).place(x=750, y=190)
button4 = tk.Button(window, image=OK, command = OKAY).place(x=670, y=250)
window.mainloop()

def Training():
    os.system('python MyCode.py')

def Testing():
    window2 = tk.Tk()
    window2.filename = filedialog.askdirectory()
    TestimagePath = window2.filename    
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    TestImage3 = image.load_img(TestimagePath, target_size=(64, 64))
    TestImage3 = image.img_to_array(TestImage3) #Converts Image to 3d Array
    TestImage3 = np.expand_dims(TestImage3, axis = 0) #converts to 4d Array
    loaded_model.predict(TestImage3)

def close():
    window3.destroy()
    
# Creating Interface to Test model
window3 = tk.Tk()
window3.geometry('850x330')
logo = tk.PhotoImage(file="UI/Logo_True.png")
Folder = tk.PhotoImage(file="UI/Test.png")
TrainLogo = tk.PhotoImage(file="UI/Train.png")
OK = tk.PhotoImage(file="UI/close.png")
label = tk.Label(window3, image= logo).place(x=5, y=10)
label2 = tk.Label(window3, text = 'Train the Model (Takes a lot of time to Train)',font = "Helvetica 13 bold").place(x=15, y=150)
button2 = tk.Button(window3, image=TrainLogo, command = Training).place(x=380, y=140)
Label3 = tk.Label(window3, text = 'Test the Model (Select an Image for Testing)',font = "Helvetica 13 bold").place(x=15, y=200)
v = StringVar() #Creating a dummy variable to change text in label
label3_1 = tk.Label(window3,anchor='w',height = 1, font='Helvetica 10',textvariable=v,relief='groove', width=45).place(x = 370, y = 200)
v.set('Please Select the working directory...')
button3 = tk.Button(window3, image=Folder, command = Testing).place(x=750, y=190)
button4 = tk.Button(window3, image=OK, command = close).place(x=635, y=250)
window3.mainloop()

    
    
    
    

