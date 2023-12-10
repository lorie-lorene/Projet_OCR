import cv2
from matplotlib import pyplot as plot
import numpy as np
import pytesseract
from PIL import Image
# ici j'affiche l'image avec tous les encombrements d'ou les trois parametres en prendre en compte
def display(im_path):
    dpi=80
    im_data=plot.imread(im_path)
    height,width,depth  =im_data.shape
    figsize=width/ float(dpi),height/ float(dpi)
    fig=plot.figure(figsize=figsize)
    ax=fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(im_data,cmap='gray')
    plot.show()
# l'image est mis en noir et blanc
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# on affiche et recupere l'mage sans les multiples couleurs
def display2(im_path):
    dpi=80
    im_data=plot.imread(im_path)
    height,width =im_data.shape
    figsize=width/ float(dpi),height/ float(dpi)
    fig=plot.figure(figsize=figsize)
    ax=fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(im_data,cmap='gray')
    plot.show()
# gestion des encombrements
def thin_font(image):
    image=cv2.bitwise_not(image)
    kernel=np.ones((1,1),np.uint8)
    image=cv2.erode(image,kernel,iterations=1 )
    image=cv2.bitwise_not(image)
    return(image)
# gestions des encombrements et mise en gras des caracteres
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
# on passe en parametre l'image qu'on veut
image_file="test2.png"
img=cv2.imread(image_file)
#cv2.imshow("orginal image",img)
#cv2.waitKey(0)

gray_image=grayscale(img)
cv2.imwrite("gray.jpg",gray_image)

#display2("gray.jpg")

thresh,im_bw=cv2.threshold(gray_image,65,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("./bw_image.jpg",im_bw)

#display2("./bw_image.jpg")

eroded_image=thin_font(im_bw)
cv2.imwrite("./eroded_image.jpg",eroded_image)

#display2("eroded_image.jpg")

dilated_image=thick_font(eroded_image)
cv2.imwrite("./dilated_image.jpg",dilated_image)

####affichage des données sur le terminal

#no_noise="dilated_image.jpg"
#img=Image.open(no_noise)
#ocr_result=pytesseract.image_to_string(img)
#print(ocr_result)
#display2("./dilated_image.jpg")

#### affichage des données dans le fichier .txt
no_noise = "dilated_image.jpg"
img = Image.open(no_noise)
ocr_result = pytesseract.image_to_string(img)

# Création du fichier et écriture des données
with open("paste.txt", "w",encoding="utf-8") as file:
    file.write(ocr_result)

print("Données extraites enregistrées dans le fichier 'paste.txt'.")