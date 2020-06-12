import cv2
import pytesseract
#read the img file
img=cv2.imread(r'/media/shivahsae/Big Data/my projects/license plate detection/car0.jpeg')
#covert into gray
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#CANNY EDGE DETECTION

canny_edge=cv2.Canny(gray,170,200)
#finding contours based on edges
contours,new=cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea, reverse=True)[:30]

#initialize license plate contour and x'y coordinates
contour_with_license_plate=None
license_plate=None
x=None
y=None
w=None
h=None

#find the contour with 4 potential corners and create ROI around it
for contour in contours:
    #find perimeters of contour and it should be a closed contour
    perimeter=cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,0.01*perimeter,True)
    if len(approx==4):
        contour_with_license_plate=approx
        x,y,w,h=cv2.boundingRect(contour)
        license_plate=gray[y:y+h,x:x+w]
        break
#removing noise from detected image

license_plate=cv2.bilateralFilter(license_plate,11,17,17)
(thresh,license_plate)=cv2.threshold(license_plate,150,180,cv2.THRESH_BINARY)

#TEXT RECOGNTION

text=pytesseract.image_to_string(license_plate)
#draw license plate and write the text
img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
img=cv2.putText(img,text,(x-100,y-50),cv2.FONT_HERSHEY_DUPLEX,6,(0,255,0),8,cv2.LINE_AA)
img=cv2.resize(img,(1000,700))
print('license_plate:',text)

cv2.imshow("license plate detector",img)
cv2.waitKey()

cv2.destroyAllWindows()

