import cv2
import numpy as np
import imutils
import pytesseract
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter as ctk

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



def recognize_license_plate(image_path):
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)


    #find contours and apply mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Find contours
    contours = imutils.grab_contours(keypoints) #Grab the contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #Sort the contours based on area and then take the largest one

    location = None #Initialize location
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break


    #mask the image
    mask = np.zeros(gray.shape, np.uint8) #Create a mask
    cv2.drawContours(mask, [location], 0,255, -1) #Draw the contours on the mask
    cv2.bitwise_and(img, img, mask=mask) #Create a new image from the mask

    #crop the image
    (x,y) = np.where(mask==255) #Find the coordinates of the contours
    (topx, topy) = (np.min(x), np.min(y)) #Find the top left corner of the contours
    (bottomx, bottomy) = (np.max(x), np.max(y)) #Find the bottom right corner of the contours
    cropped = gray[topx:bottomx+1, topy:bottomy+1] #Crop the image

 
    text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 6')
    print(text)

    cv2.putText(img, text, (topy,topx- 10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2) #Draw the text on the image
    cv2.rectangle(img, (topy, topx), (bottomy, bottomx), (0,0,255),3) #Draw a rectangle around the image
    
    return img


def select_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        processed_image = recognize_license_plate(file_path)
        display_image(processed_image)


def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(img) 
    label.configure(image=img_tk)


Window = ctk.CTk()
Window.title("License Plate Recognition")
Window.geometry("800x600")


myframe = ctk.CTkScrollableFrame(Window)
myframe.pack(fill = "both" , expand = True)

label = ctk.CTkLabel(myframe, text=" ")
label.pack()

select_button = ctk.CTkButton(myframe, text="Select Image", command=select_image)
select_button.pack()

Window.mainloop()



