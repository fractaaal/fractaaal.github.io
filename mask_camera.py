import cv2
import numpy as np

cap = cv2.VideoCapture(0)
frx = 1000
fry = 600

while True:
    #前景の動画を読み込み
    _, frame = cap.read()
    frame = cv2.resize(frame, (frx,fry))
    #背景画像の読み込み
    back_img = cv2.imread("hawai.jpg")
    back_img = cv2.resize(back_img, (frx, fry))
    
    #グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7),0)
    frame_front = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    #輪郭抽出
    cnts_front = cv2.findContours(frame_front,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[0]
    
    #輪郭内を白で塗り潰し
    mask_front = np.zeros_like(frame_front)
    cv2.drawContours(mask_front, cnts_front, -1, color=255, thickness=-1)
   
    #反転する
    #輪郭内に前景画内を描写し、輪郭外は白に塗りつぶし
    mask_front = cv2.cvtColor(mask_front,cv2.COLOR_GRAY2BGR)
    mask_front_inverse = cv2.bitwise_not(mask_front)
    masked_front_upstate = cv2.bitwise_and(frame,mask_front)
    masked_front_white = cv2.addWeighted(masked_front_upstate, 1, \
    mask_front_inverse,1,0)                           
    
    #背景画像と反転した画像のandを返す
    #合成
    masked_back_upstate = cv2.bitwise_and(back_img,mask_front_inverse)
    masked_back_white = cv2.addWeighted(masked_back_upstate, 1, \
    mask_front,1,0) 
        
    result = cv2.bitwise_and(masked_front_white,masked_back_white)
    
    cv2.imshow("Mask Camera", result)
    if cv2.waitKey(1) == 13: break
cap.release()
cv2.destroyAllWindows()
