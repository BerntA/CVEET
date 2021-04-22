import os
import time
import cv2
import numpy as np
import requests as req
import datetime as dt

URLS = [
    #('e39_aug_s', 'https://www.vegvesen.no/public/webkamera/kamera?id=2986247', cv2.IMREAD_COLOR), # E39 Auglend Sør
    #('e39_aug_n', 'https://www.vegvesen.no/public/webkamera/kamera?id=2992499', cv2.IMREAD_COLOR), # E39 Auglend Nord
    #('e6_osen', 'https://www.vegvesen.no/public/webkamera/kamera?id=100082', cv2.IMREAD_COLOR), # E6 Osen
    #('fv44_kvassheim', 'https://www.vegvesen.no/public/webkamera/kamera?id=125196', cv2.IMREAD_COLOR), # FV44 Kvassheim mot Sandnes
    #('e39_vatland', 'https://www.vegvesen.no/public/webkamera/kamera?id=458576', cv2.IMREAD_COLOR), # E39 Vatland
    #('e6_nostvet', 'https://www.vegvesen.no/public/webkamera/kamera?id=2580261', cv2.IMREAD_COLOR), # E6 Nøstvet
    #('rv150_ullevaal', 'https://www.vegvesen.no/public/webkamera/kamera?id=410395', cv2.IMREAD_COLOR), # Riksvei 150 Ullevål
    ('rv509_sommevaagen', 'https://www.vegvesen.no/public/webkamera/kamera?id=751511', cv2.IMREAD_COLOR), # Nærme Sola.
]

IMG_HTBL = set() # Store image hashes here so we do not re-save the same images over and over again.

def get_image(prefix, url, flags=cv2.IMREAD_COLOR, folder='raw', quality=90):    
    r = req.get(url)
    if r.status_code != 200: # Is the camera down?
        return None
    
    h = hash(str(r.content))
    if h in IMG_HTBL: # Don't process the same image twice.
        return None
    
    IMG_HTBL.add(h)
    img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), flags)
    img = img[250:,:,:] # Exclude the top black bar.
    timestr = dt.datetime.now().strftime("%d-%m-%H-%M")
    cv2.imwrite('../images/{}/{}_{}.jpg'.format(folder, prefix, timestr), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    print("Wrote IMG -", '{}/{}_{}.jpg'.format(folder, prefix, timestr))

while True:
    for prefix, url, flags in URLS:
        _ = get_image(prefix, url, flags, 'raw_night', 100)
        time.sleep(0.2)
    time.sleep(30)
