#still image prototype SPI-NET with GUI

from tkinter import *
import tkinter #library files for GUI
import tkinter.scrolledtext
from PIL import Image, ImageTk #library files for still images
import cv2 #OpenCv for camera feed
import os #Creating a folder


# prompt: neural network for identifying faces using pytorch

#module.py
# prompt: neural network for identifying faces using pytorch
#this code is used for training the NN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#Main Menu GUI

class MainScreen():
    def __init__(self):
        self.frame = Frame(window)
        self.frame.pack() 
        self.label = Label(self.frame, text='Main Menu')
        self.label.pack()
        add_button = Button(self.frame, text="Add Person", command=self.addScreen)
        add_button.pack()
        live_button = Button(self.frame, text="Live Camera", command=self.liveScreen)
        live_button.pack()
        Test_button = Button(self.frame, text="Test Person", command=self.testScreen)
        Test_button.pack()
        
    def addScreen(self):
        self.frame.pack_forget()
        self.add_screen=AddScreen(parent=self)
        self.add_screen.startAddScreen()
    
    def liveScreen(self):
        self.frame.pack_forget()
        self.live_screen=LiveFeedScreen(parent=self)
        self.live_screen.startLiveScreen()

    def mainScreen(self):
        self.frame.pack()

    def testScreen(self):
        self.frame.pack_forget()
        self.test_screen=TestScreen(parent=self)
        self.test_screen.startTestScreen()
    
#Test Person GUI
class TestScreen():
    def __init__(self,parent=None):
        self.menu_screen=parent
        self.frame=Frame(window, bg="green")
        self.label= Label(self.frame, text='Selfie Time')
        self.camera_label = Label(self.frame, bg="blue")
        self.Camera = CameraFunctions(self.camera_label)
        self.label.pack()
        self.camera_label.pack()
        self.name_label = Label(self.frame, text="Enter Name: ")
        self.name_label.pack()
        self.name_input_box=Entry(self.frame)
        self.name_input_box.pack()
        self.capture_frame_button = Button(self.frame, text="Capture Images", command= self.captureSetup)
        self.capture_frame_button.pack()
        self.test_img_button = Button(self.frame, text="Test Images", command= self.callTestFunction)
        self.test_img_button.pack()
        self.menu_button = Button(self.frame, text="Menu", command=self.menu).pack()
        self.exit_button = Button(self.frame, text="Exit", command=self.exit).pack()

    def startTestScreen(self):
        self.frame.pack()
        self.Camera.capture=True
        self.Camera.showFrames()

    def callTestFunction(self):
        self.Camera.closeCamera()
        SPINET = TestWithImage(self.camera_label)

    def captureSetup(self):

        name = self.name_input_box.get()
        name.replace(" ","")
        name.lower()
        self.Camera.captureTestFrames(name)



    def menu(self):
        self.Camera.capture=False
        self.Camera.closeCamera()
        self.frame.pack_forget()
        self.menu_screen.mainScreen()

    def exit(self):
        window.quit()

class AddScreen():
    def __init__(self,parent=None):
        self.menu_screen=parent
        self.frame=Frame(window, bg="green")
        self.label= Label(self.frame, text='Selfie Time')
        self.camera_label = Label(self.frame, bg="blue")
        self.label.pack()
        self.camera_label.pack()
        self.name_label = Label(self.frame, text="Enter Name: ")
        self.name_label.pack()
        self.name_input_box=Entry(self.frame)
        self.name_input_box.pack()
        self.Camera = CameraFunctions(self.camera_label)

        self.name_error_label = Label(self.frame)

        self.capture_frame_button = Button(self.frame, text="Capture Images", command= self.captureSetup)
        self.capture_frame_button.pack()
        self.train_button = Button(self.frame, text="Train Model", command=self.trainScreen)
        self.train_button.pack()
        self.menu_button = Button(self.frame, text="Menu", command=self.menu).pack()
        self.exit_button = Button(self.frame, text="Exit", command=self.exit).pack()

    def startAddScreen(self):
        self.frame.pack()
        self.Camera.showFrames()


    def captureSetup(self):
        name = self.name_input_box.get()
        if (name==""):
            self.name_error_label.config(text="Enter a name")
            self.name_error_label.pack()
        else:
 #DO MORE NAME CONVENTIONS HERE?
            name.replace(" ","")
            name.lower()
            self.Camera.captureFrames(name)
            self.changeButtons()
    
    def changeButtons(self):
        self.name_error_label.pack_forget()
        self.name_input_box.pack_forget()
        self.name_label.pack_forget()
        self.capture_frame_button.pack_forget()
        self.train_button.pack()

            
    def trainScreen(self):
        self.frame.pack_forget()
        self.train_screen=TrainScreen(parent=self)
        self.train_screen.startTrainScreen()

    def menu(self):
       
        self.Camera.closeCamera()
        self.frame.pack_forget()
        self.menu_screen.mainScreen()

    def exit(self):
        window.quit()

#View Live Feed GUI

class LiveFeedScreen():
     def __init__(self,parent=None):
        self.menu_screen=parent
        self.frame=Frame(window, bg="blue")
        self.label= Label(self.frame, text='View Camera')
        self.label.pack()    

     def startLiveScreen(self):
        self.componenets()
        self.frame.pack()

     def componenets(self):
        menu_button = Button(self.frame, text="Menu", command=self.menu).pack()
        exit_button = Button(self.frame, text="Exit", command=self.exit).pack()

     def menu(self):   
        self.frame.pack_forget()
        self.menu_screen.mainScreen()

     def exit(self):
        window.quit()


#View training information

#Add Person GUI
class TrainScreen():
    def __init__(self,parent=None):
        self.menu_screen=parent
        self.frame=Frame(window, bg="green")
        self.label= Label(self.frame, text='Training Time')
        self.label.pack()
        self.training_text = tkinter.scrolledtext.ScrolledText(self.frame)
        self.training_text.pack()

        self.Training = TrainWithImage(self.training_text)

    def componenets(self):
        menu_button = Button(self.frame, text="Menu", command=self.menu).pack()
        exit_button = Button(self.frame, text="Exit", command=self.exit).pack()


    def startTrainScreen(self):
        self.frame.pack()
        self.Training.startTraining()
        self.componenets()

    def menu(self):
        self.frame.destroy()
        self.menu_screen.menu()

    def exit(self):
        window.quit()

    

#Camera Functions

class CameraFunctions():
    def __init__(self, label=None):
        self.camera_label=label
        self.cap= cv2.VideoCapture(0)
        self.capOn=True
        self.addPerson=False
        self.capture=True
#Open Camera
    def closeCamera(self):
        self.capture=False
        self.cap.release()
        cv2.destroyAllWindows()
    def showFrames(self):
        #pauses whilst not on screen

        #Image size 640x640
        if (self.capture):
            # Get the latest frame and convert into Image
            self.frame= cv2.cvtColor(self.cap.read()[1],cv2.COLOR_BGR2RGB)
            #Crop image [y:y+h, x:x+w]
            self.frame = self.frame[30:280, 10:280]

            #Cannot grey scale for viewing, PhotoImage needs 3 channels
            self.img = Image.fromarray(self.frame)
            # Convert image to PhotoImage
            self.imgtk = ImageTk.PhotoImage(image = self.img)
            self.camera_label.imgtk = self.imgtk
            self.camera_label.configure(image=self.imgtk)
        
            # Repeat after an interval to capture continiously
            self.camera_label.after(20, self.showFrames)


#Capture Frames
    def captureFrames(self,name_input=""):
        #INITIALISE value to count frames
        frameNumber=0
        name=name_input
        #INITIALISE value to SET frame save name
        imageNumber=0
        
        trainpath = "./train"

        if not os.path.exists(trainpath):
            os.mkdir(trainpath)


        #Create a folder
        path = f"./{trainpath}/"+name
        
        if not os.path.exists(path):
            os.mkdir(path)

        while True:
          
#CANNOT GET FRAME TO UPDATE WHILST CAPTURING
            #self.showFrames()
            frameNumber=frameNumber+1
        # Capture frame-by-frame
            ret, frame = self.cap.read()
 
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        #Crop image [y:y+h, x:x+w]
            frameCropped = frame[80:280, 50:280]

        #Gray scale
            frameGrayScale = cv2.cvtColor(frameCropped, cv2.COLOR_BGR2GRAY)

        #Display frame change to frameCropped if not wanting gray scale
            #cv2.imshow('frame', frameGrayScale)
        
        #Save image file
            if (frameNumber%10==0):
                
            #SET image number
                imageNumber=imageNumber+1
            #CHANGE to your chosen path
                file_name = f"./{trainpath}/"+name+"/"+name+str(imageNumber)+'.jpg'
                cv2.imwrite(file_name,frameGrayScale)
                if(imageNumber==100):
                    break


    def captureTestFrames(self,name_input=""):
        name=name_input
        #INITIALISE value to count frames
        frameNumber=0
        #INITIALISE value to SET frame save name
        imageNumber=0
        
        testpath = "./test"

        if not os.path.exists(testpath):
            os.mkdir(testpath)


        #Create a folder"
        path = f"./{testpath}/{name}"
        
        if not os.path.exists(path):
            os.mkdir(path)

        while True:
          
            frameNumber=frameNumber+1
        # Capture frame-by-frame
            ret, frame = self.cap.read()
 
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
        #Crop image [y:y+h, x:x+w]
            frameCropped = frame[80:280, 50:280]

        #Gray scale
            frameGrayScale = cv2.cvtColor(frameCropped, cv2.COLOR_BGR2GRAY)

        #Display frame change to frameCropped if not wanting gray scale
            #cv2.imshow('frame', frameGrayScale)
        
        #Save image file
            if (frameNumber%10==0):
                
            #SET image number
                imageNumber=imageNumber+1
            #CHANGE to your chosen path
                file_name = f"./{testpath}/"+name+"/"+name+str(imageNumber)+'.jpg'
                cv2.imwrite(file_name,frameGrayScale)
                if(imageNumber==20):
                    break

#Retrain Image recognition

# Define the neural network architecture
class FaceRecognitionNet(nn.Module):
    def __init__(self):
        super(FaceRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # parameters are input channels, output channels, kernel size , padding  
        self.relu1 = nn.ReLU() # convert negative values to zero
        self.pool1 = nn.MaxPool2d(2, 2) # takes the highest values within a 2x2 window to highlight most prominent features
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # next input recieves output channels of previous of conv2d function
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 100 * 100, 512) # Adjust input size based on image dimensions
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 128 * 100 * 100) # Adjust input size based on image dimensions
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

#Face Recognition

class TrainWithImage():

    def __init__(self, text=None):
        self.training_text=text

    def startTraining(self):
        # Set device (GPU if available)
        device = torch.device("cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Hyperparameters
        learning_rate = 0.001
        num_epochs = 5
        batch_size = 30

        # Data transformations and loading (replace with your actual dataset)
        transform = transforms.Compose([
            transforms.Resize((400, 400)),  # Adjust image size as needed
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.ImageFolder(root="./train/", transform=transform) # Replace with your dataset path
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        # Initialize the model, loss function, and optimizer
        model = FaceRecognitionNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(trainloader):
                images = images.to(device) # passes images to GPU/CPU
                labels = labels.to(device) # passses labels to GPU/CPU

                outputs = model(images)  # forward pass of NN
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward() # Backward pass of NN
                optimizer.step()

                #if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}")
                add_text=f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}"
                self.training_text.insert(END,add_text+"\n")

        
        #print("Training finished")
        self.training_text.insert(END,"Training finished")

        # Save the model
        torch.save(model.state_dict(), "face_recognition_model300final.pth")


class TestWithImage():
     def __init__(self, label=None):
        self.camera_label=label
        self.testimage()    


     def testimage(self):
        device = torch.device("cpu")
        # Load the model and set it to evaluation mode
        model = FaceRecognitionNet()
        model.load_state_dict(torch.load("face_recognition_model300final.pth"))
        model.eval()

        # Test transformations (same as training transformations)
        transform = transforms.Compose([
            transforms.Resize((400, 400)),  # Adjust image size as needed
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load test dataset (replace with your actual test dataset path)
        testset = datasets.ImageFolder(root="./test", transform=transform) # Replace with your test dataset path
        testloader = DataLoader(testset, batch_size=10, shuffle=False)
        
        # Test the model
        correct = 0
        total = 0
        incorrect = 0
        incorrectarr = [] 
        with torch.no_grad():
            for index,(images, labels) in enumerate(testloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                
                incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0] 

                for idx in incorrect_indices:
                    print(f"Incorrect Prediction: Predicted={predicted[idx].item()}, Actual={labels[idx].item()}") # print out results of incorrect predictions
                    incorrectarr.append(f"Batch: {index} Image: {idx.item()}")
                correct += (predicted == labels).sum().item()
                incorrect += (predicted != labels).sum().item()
                
                print(f'person = {labels}')#batch of images    
                print(f'correct = {correct}') # how many got correct
                print(f'total = {total}') # total images looked at
                print(f'incorrect prediction = {incorrect} against {total}')  # total incorrect
        

        print(incorrectarr) #print array of which images were incorrect and from which batch.
        percentage = 100 * correct / total

        if (percentage>=75):
            pathFile="granted"
        else:
            pathFile="denied"
        
        path = f"./access_images/"+pathFile+'.png'
        img = ImageTk.PhotoImage(Image.open(path))
        self.camera_label.configure(image=img)
        self.camera_label.image=img
        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')




#Start Application
window=Tk() #Create a window
window.geometry("500x600") ##set sizes
window.resizable(False,False) #stop resizeable
main = MainScreen() #create main menu object
window.mainloop() #Executes application until exit called



#pip install cv
#pip install opencv-python
#pip install pillow
#pip install numpy
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124