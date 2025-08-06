# 실습
```python
import cv2
import numpy as np

categories = ['dont_enter', 'dont_left', 'dont_right', 'limit']
dict_file = './traffic_dict.npy'
svm_model_file = './traffic_svm.xml'

# 테스트할 이미지
imgs = ['../img/dont_enter.png','../img/dont_left.png',
        '../img/dont_right.png','../img/limit.png','../img/enter.jpg']

detector = cv2.xfeatures2d.SIFT_create()
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, cv2.BFMatcher(cv2.NORM_L2))
bowExtractor.setVocabulary(np.load(dict_file))
svm = cv2.ml.SVM_load(svm_model_file)

for i, path in enumerate(imgs):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = bowExtractor.compute(gray, detector.detect(gray))
    ret, result = svm.predict(hist)
    name = categories[int(result[0][0])]
    txt, base = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 2, 3)
    x, y = 10, 50
    
    scale = 3.0
    resize = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    cv2.rectangle(resize, (x, y-base-txt[1]), (x+txt[0], y+txt[1]), (30,30,30), -1)
    cv2.putText(resize, name, (x,y), cv2.FONT_HERSHEY_PLAIN,
                2, (0,255,0),2, cv2.LINE_AA)
    cv2.imshow(name, resize)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# 실행결과
<img width="602" height="659" alt="resulttttttt" src="https://github.com/user-attachments/assets/86a00dac-aaf9-4e21-8c7f-1c8b3351b785" />
