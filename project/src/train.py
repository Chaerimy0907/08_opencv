import cv2
import numpy as np
import os, glob

# 카테고리 설정
categories = ['dont_enter', 'dont_left', 'dont_right', 'limit']
#dictionary_size = 100  # 시각 사전 크기 (조정 가능)
dictionary_size = 50
base_path = "../data/"  # 데이터셋 경로 (project/src 기준)
dict_file = './traffic_dict.npy'
svm_model_file = './traffic_svm.xml'

# SIFT + BOW 준비
detector = cv2.xfeatures2d.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)
bowTrainer = cv2.BOWKMeansTrainer(dictionary_size)
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, matcher)

# 사전 구축
train_paths = []
train_labels = []
print("[1] 특징 디스크립터 수집 중...")
for idx, category in enumerate(categories):
    img_paths = glob.glob(os.path.join(base_path, category, '*.jpg'))
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kpt, desc = detector.detectAndCompute(gray, None)
        if desc is not None:
            bowTrainer.add(desc)
            train_paths.append(img_path)
            train_labels.append(idx)
    print(f"\t{category} 완료 ({len(img_paths)} 장)")

print("[2] KMeans로 시각 사전 생성 중...")
dictionary = bowTrainer.cluster()
np.save(dict_file, dictionary)
bowExtractor.setVocabulary(dictionary)
print("시각 사전 생성 완료:", dictionary.shape)

# 학습 데이터 BOW 변환
train_desc = []
for path in train_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = bowExtractor.compute(gray, detector.detect(gray))
    if hist is not None:
        train_desc.extend(hist)

print(f"[3] SVM 학습 데이터 크기: {len(train_desc)}")

# SVM 학습
svm = cv2.ml.SVM_create()
svm.trainAuto(np.array(train_desc), cv2.ml.ROW_SAMPLE, np.array(train_labels))
svm.save(svm_model_file)
print("[4] SVM 학습 완료, 모델 저장됨:", svm_model_file)
