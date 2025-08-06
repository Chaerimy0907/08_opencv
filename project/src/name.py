import os

# 데이터가 있는 루트 폴더
data_dir = "../data"

# 카테고리별 폴더 순회
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        continue

    for idx, filename in enumerate(os.listdir(category_path)):
        ext = os.path.splitext(filename)[1]  # 확장자 (.jpg 등)
        new_name = f"{category}_{idx+1}{ext}"
        old_path = os.path.join(category_path, filename)
        new_path = os.path.join(category_path, new_name)
        os.rename(old_path, new_path)

    print(f"[완료] {category} 폴더 파일명 정리")