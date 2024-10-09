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

vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)


frame_width = 1024
frame_height = 576


def send_an_email():
    print('sending mail')
    toaddr = 'vamsijayanth5@gmail.com'      # To id 
    me = 'vamsijayanth5@gmail.com'          # your id
    subject = "Baby_image"              # Subject
  
    msg = MIMEMultipart()  
    msg['Subject'] = subject  
    msg['From'] = me  
    msg['To'] = toaddr  
    msg.preamble = "test "   
    #msg.attach(MIMEText(text))  
  
    part = MIMEBase('application', "octet-stream")  
    part.set_payload(open("image.jpg", "rb").read())  
    encoders.encode_base64(part)  
    part.add_header('Content-Disposition', 'attachment; filename="image.jpg"')   # File name and format name
    msg.attach(part)  
  
    try:  
       s = smtplib.SMTP('smtp.gmail.com', 587)  # Protocol
       s.ehlo()  
       s.starttls()  
       s.ehlo()  
       s.login(user = 'vamsijayanth5@gmail.com', password = 'Saivamsi@01')  # User id & password
       #s.send_message(msg)  
       s.sendmail(me, toaddr, msg.as_string())  
       s.quit()  
    #except:  
    #   print ("Error: unable to send email")    
    except SMTPException as error:  
          print ("Error")                # Exception


          
 
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

#ii=0
ests=1
while True:
    frame = vs.read()
#    ii=ii+1
#    if(ii>50):
#        ii=0
#        cv2.imwrite('image.jpg',frame)
#        cv2.waitKey(1)
#        time.sleep(0.5)
#        send_an_email()

    abnormal=0
    normal=0
    val = (hx.get_weight(5))/500000
    wgt=round(val,2)
    
    tmp=read_temp()
    print("body temparature:"+str(tmp)+" F")
    print("baby weight:"+str(wgt)+"Kg")
    if(wgt<0.1):
        print("No Baby")
        bsts='no_baby'
        #abnormal=1
    else:
        normal=normal+1
        bsts='baby_present'
        
    if(tmp>99):
        print("HIGH Body Temperature")
        abnormal=1
    else:
        normal=normal+1
        
    if(GPIO.input(ss)==1):
        print("sound_detected:")
        snds="sound_detected"
        abnormal=1
        snd=1
    else:
        print("no sound detected:")
        snds="sound_NOT_detected"
        snd=0
        normal=normal+1

    if(normal==3):
        ests=1
        
    if(abnormal==1 and ests==1):
        cv2.imwrite('image.jpg',frame)
        cv2.waitKey(1)
        time.sleep(0.5)
        ests=0
        send_an_email()

    wp = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=G0ABY4U28CZN1T6R&field1=" + str(tmp) + "&field2=" + str(wgt) + "&field3=" + str(snds) + "&field4=" + str(bsts))
    cv2.imshow('video',frame)
    cv2.waitKey(1)
    

