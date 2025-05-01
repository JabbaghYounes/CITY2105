from tkinter import * #library files for GUI
from PIL import Image, ImageTk #library files for still images
import cv2 #OpenCv for camera feed
import os #Creating a folder

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

#Add Person GUI
class AddScreen():
    def __init__(self,parent=None):
        self.menu_screen=parent
        self.frame=Frame(window, bg="green")
        self.label= Label(self.frame, text='Selfie Time')
        self.CameraLabel = Label(self.frame, bg="blue")
        self.label.pack()
        self.CameraLabel.pack()
        Label(self.frame, text="Enter Name: ").pack()
        self.name_input_box=Entry(self.frame)
        self.name_input_box.pack()
        self.Camera = CameraFunctions(self.CameraLabel)



    def startAddScreen(self):
        self.componenets()
        self.frame.pack()
        self.Camera.showFrames()

    def componenets(self):
        capture_frame_button = Button(self.frame, text="Capture Images", command= self.captureSetup).pack()
        menu_button = Button(self.frame, text="Menu", command=self.menu).pack()
        exit_button = Button(self.frame, text="Exit", command=self.exit).pack()

    def captureSetup(self):
        name = self.name_input_box.get()
        self.Camera.captureFrames(name)


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

#Camera Functions

class CameraFunctions():
    def __init__(self, label=None):
        self.camera_label=label
        self.cap= cv2.VideoCapture(0)
        self.capOn=True
        self.addPerson=False
#Open Camera
    def closeCamera(self):
        self.cap.release()
    def showFrames(self):
        # Get the latest frame and convert into Image

        frame= cv2.cvtColor(self.cap.read()[1],cv2.COLOR_BGR2RGB)
        #Crop image [y:y+h, x:x+w]
        frame = frame[80:280, 50:280]
        #Cannot grey scale for viewing, PhotoImage needs 3 channels
        img = Image.fromarray(frame)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        # Repeat after an interval to capture continiously
        self.camera_label.after(20, self.showFrames)
#Capture Frames
    def captureFrames(self,name_input=""):
        #INITIALISE value to count frames
        frameNumber=0
        name=name_input
        #INITIALISE value to SET frame save name
        imageNumber=0

        #Create a folder
        path = "./"+name

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
                file_name = "./"+name+"/"+name+str(imageNumber)+'.jpg'
                cv2.imwrite(file_name,frameGrayScale)
                if(imageNumber==100):
                    break

#Retrain Image recognition

#Face Recognition

#Start Application
window=Tk() #Create a window
window.geometry("500x600") ##set sizes
window.resizable(False,False) #stop resizeable
main = MainScreen() #create main menu object
window.mainloop() #Executes application until exit called

