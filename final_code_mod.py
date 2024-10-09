import os
import glob
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
import imutils
import time
from imutils.video import VideoStream
import smtplib,ssl
from email.mime.multipart import MIMEMultipart  
from email.mime.base import MIMEBase  
from email.mime.text import MIMEText  
from email.utils import formatdate  
from email import encoders
import urllib.request
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image

EMULATE_HX711=False

referenceUnit = 1

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711
def cleanAndExit():
    print("Cleaning...")

    if not EMULATE_HX711:
        GPIO.cleanup()
        
    print("Bye!")
    sys.exit()

hx = HX711(26, 19)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(referenceUnit)

hx.reset()

hx.tare()

print("Tare done! Add weight now...")


model = tf.keras.models.load_model('./model.h5')

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False) 
os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')
 
base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'
ss=2
GPIO.setup(ss,GPIO.IN)
print("[INFO] initializing camera...")
bg = None
bsts='Capturing'
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    invert = cv2.bitwise_not(thresholded)
    cv2.imshow("Thesholded", invert)
    cv2.waitKey(1)
    cv2.imwrite('test.jpg',invert)
    cv2.waitKey(1)
    time.sleep(0.2)

    clssify()   

def probToClass(proList):
  postures = ['lying', 'sitting', 'No baby']
  return postures[np.argmax(proList)]
 
def clssify():
    global bsts

    filelist = [ f for f in os.listdir() if f.endswith('.jpg') ]

    path = '/home/pi/Desktop/baby monitor/test.jpg'
    img = image.load_img(path, target_size=(255, 255), color_mode="grayscale" )
    imag=cv2.imread('test.jpg')
    time.sleep(0.1)
    number_of_black_pix = np.sum(imag == 0)
    print(number_of_black_pix)
    if(number_of_black_pix<100000):
        print('No baby')
        bsts="No_Baby"
    else:
        x = image.img_to_array(img)
        x = x*(1/255)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        print(probToClass(classes))
        bsts=probToClass(classes)
    time.sleep(3)

def send_an_email():
    print('sending mail')
    toaddr = 'vamsijayanth5@gmail.com'      
    me = 'vamsijayanth5@gmail.com'          
    subject = "Captured"              
  
    msg = MIMEMultipart()  
    msg['Subject'] = subject  
    msg['From'] = me  
    msg['To'] = toaddr  
    msg.preamble = "test "   
     
    part = MIMEBase('application', "octet-stream")  
    part.set_payload(open("image.jpg", "rb").read())  
    encoders.encode_base64(part)  
    part.add_header('Content-Disposition', 'attachment; filename="image.jpg"')
    msg.attach(part)  
  
    try:  
       s = smtplib.SMTP('smtp.gmail.com', 587) 
       s.ehlo()  
       s.starttls()  
       s.ehlo()  
       s.login(user = 'vamsijayanth5@gmail.com', password = '***********')
       s.sendmail(me, toaddr, msg.as_string())  
       s.quit()   
    except SMTPException as error:  
          print ("Error")                         
 
def read_temp_raw():
    f = open(device_file, 'r')
    lines = f.readlines()
    f.close()
    return lines
 
def read_temp():
    lines = read_temp_raw()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos+2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return  temp_f
ii=0

aWeight = 0.5

top, right, bottom, left = 100, 100, 400, 600

num_frames = 0
start_recording = False
ii=0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 700)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    if num_frames < 30:
        run_avg(gray, aWeight)

        
    else:
        ii=ii+1
        if(ii>10):
            ii=0
            #hand = segment(gray)

        val = ((hx.get_weight(5))/500000)
        val=round(val,2)           
        wgt=val
        tmp=read_temp()
        print("body temparature:"+str(tmp)+" F")

        print("baby weight:"+str(wgt)+"Kg")


        if(tmp>100):
            hand = segment(gray)
            cv2.imwrite('image.jpg',frame)
            cv2.waitKey(1)
            send_an_email()


            
        if(GPIO.input(ss)==1):
            print("sound_detected:")
            snds="sound_detected"
            hand = segment(gray)
            cv2.imwrite('image.jpg',frame)
            cv2.waitKey(1)
            
            send_an_email()
            snd=1
        else:
            print("no sound detected:")
            snds="sound_NOT_detected"
            snd=0

        print('baby status:' + bsts)

        wp = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=G0ABY4U28CZN1T6R&field1=" + str(tmp) + "&field2=" + str(wgt) + "&field3=" + str(snds) + "&field4=" + str(bsts))


    cv2.imshow("frame", clone)
    cv2.waitKey(1)
    num_frames += 1

