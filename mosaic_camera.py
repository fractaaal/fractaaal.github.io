import cv2

cap = cv2.VideoCapture(0)
frx = 1000
fry = 600

cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

def mosaic(img, rect, size):
    #モザイクをかける領域の取得
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 -y1
    i_rect = img[y1:y2, x1:x2]
    
    #一度縮小して拡大する
    i_small = cv2.resize(i_rect, ( size, size))
    i_mos = cv2.resize(i_small, (w, h),  interpolation=cv2.INTER_AREA)
    
    #画像にモザイクをかける
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2

while True:
    #前景の動画を読み込み
    _,frame = cap.read()
    frame = cv2.resize(frame, (frx,fry))
    
    #グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    #顔検出を実行
    face_list = cascade.detectMultiScale(gray, minSize=(100,100))
    
    if len(face_list)==0: continue

        #認識した部分の画像にモザイクをかける
    for (x,y,w,h) in face_list:
        result = mosaic(frame, (x,y,x+w, y+h),10)
    
    cv2.imshow("mosaic",result)
    if cv2.waitKey(1) == 13: break
cap.release()
cv2.destroyAllWindows()
