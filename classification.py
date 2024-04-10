import pandas as pd
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Input
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix
import shutil
from datetime import datetime
from PIL import Image
import time


def get_date_made(path):
    # 파일 경로 지정
    file_path = 'C:/Users/MING/Desktop/sdfa/test_image'

    # 파일의 생성 시간을 초 단위의 타임스탬프로 가져옴
    creation_time_stamp = os.path.getctime(file_path)

    # 타임스탬프를 datetime 객체로 변환
    creation_time = datetime.fromtimestamp(creation_time_stamp)

    # 날짜와 시간을 보기 쉬운 형식의 문자열로 출력
    formatted_time = creation_time.strftime('%Y:%m:%d %H:%M:%S')

    return formatted_time


def get_date_taken(path):
    # 이미지 열기
    image = Image.open(path)

    # 이미지에서 Exif 데이터 가져오기
    exif_data = image._getexif()

    # Exif 데이터가 없거나 'DateTimeOriginal' 키가 없는 경우 None 반환
    if not exif_data:
        return None

    # Exif 태그 중 DateTimeOriginal 태그는 찍은 날짜와 시간을 담고 있습니다 (태그 ID는 36867)
    date_taken = exif_data.get(36867)

    return date_taken


image_dir = Path('./test_image')
# Get filepaths and labels
# .glob() will return all file paths that match a specific pattern
filepaths = list(image_dir.glob(r'**/*.jpeg'))


def extract_label(filepath):
    filename = filepath.name
    # ' ' (공백)을 기준으로 문자열을 나누고 첫 번째 부분을 가져옴 (예: 'plastering')
    label = filename.split(' ')[0]
    return label


labels = list(map(extract_label, filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
# Concatenate filepaths and labels --> We will obtained a labeled dartaset
image_df = pd.concat([filepaths, labels], axis=1)

# Drop GT images
image_df = image_df[image_df['Label'].apply(lambda x: x[-2:] != 'GT')]

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_image = test_generator.flow_from_dataframe(
    dataframe=image_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=len(image_df['Label']),
    shuffle=False
)

model_yolo = YOLO('predict_model.pt')
results = model_yolo.predict(source='test_image', save=True)
image_path = [results[i].path for i in range(93)]
result_df = pd.DataFrame(image_path, columns=['result_image_path'])

tfidf = pd.read_csv('tfidf_table.csv', encoding='utf-8-sig', index_col=0)
tfidf_tensor = tf.convert_to_tensor(tfidf)

# merge object
truck = ['dump truck', 'mixed']
plastering_tool = ['Trowel', 'rainboot', 'plasterer shoe']
brick = ['brick', 'bricks']
concrete = ['concrete bag', 'basin']
liquid_material = ['plastic barrel', 'square can']
merge_lst = [truck, plastering_tool, brick, concrete, liquid_material]
merge_lst_name = ['truck', 'plastering_tool', 'brick', 'concrete', 'liquid_material']

# 탐지 객체 개수 저장
result_lst = []
for result in results:
    # 결과를 저장할 딕셔너리
    result_dict = {}
    result_dict = {i: 0 for i in range(len(model_yolo.names))}

    # 텐서에서 나오는 클래스 별로 카운트
    for val in result.boxes.cls:
        result_dict[int(val)] += 1

    # 딕셔너리를 데이터프레임으로 변환
    df = pd.DataFrame(list(result_dict.items()), columns=['Class', 'Count'])
    df['Class_name'] = model_yolo.names.values()

    result_lst.append(df)

# 객체 카테고리별 병합
merge_object_lst0 = []
for df in result_lst:
    for category in merge_lst:
        count = 0
        for object in category:
            count += df['Count'][df.loc[df['Class_name'] == object].index[0]]
            df = df.drop(df.loc[df['Class_name'] == object].index[0])
        new_row = pd.DataFrame(
            {'Class': [len(df)], 'Count': [count], 'Class_name': [merge_lst_name[merge_lst.index(category)]]})
        df = pd.concat([df, new_row], axis=0, ignore_index=True)
    df = df.sort_values('Class_name')
    df = df.drop(['Class', 'Class_name'], axis=1).reset_index(drop=True)
    tensor = tf.squeeze(tf.convert_to_tensor(df.reset_index(drop=True), dtype=tf.float32))
    merge_object_lst0.append(tensor)

# 객체 개수*tfidf테이블
merge_object_lst1 = []
for result in merge_object_lst0:
    var = tf.Variable(tfidf_tensor)
    for cls in range(0, len(tfidf.columns)):
        # 탐지된 객체의 개수가 0이면 tfidf점수를 0으로 변환, cls는 객체의 인덱스
        if result[cls] == 0:
            for i in range(6):
                var[i, cls].assign(0)
        # 탐지된 객체의 개수가 0초과면 tfidf점수를 객체의 개수만큼 곱함
        else:
            for i in range(6):
                var[i, cls].assign(var[i, cls] * tf.cast(result[cls], tf.float64))
    merge_object_lst1.append(tf.reshape(var, [-1]))

object_array = np.array(merge_object_lst1)
image_array, image_label = next(test_image)

# lagorithm - neural network - deep learning
classify_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

classify_model.trainable = False

# 1280+84
image_input = classify_model.input  # 이미지
object_input = Input(shape=(84,))  # lst_merge[0].input #객체 탐지, 표준시방서 데이터(84,)
output1 = tf.keras.layers.Concatenate(axis=1)([classify_model.output, object_input])  # (1280+90,)
dropout = tf.keras.layers.Dropout(rate=0.2)(output1)

x = tf.keras.layers.Dense(128, activation='relu')(dropout)
x = tf.keras.layers.Dense(128, activation='relu')(x)  # relu is the activation function for neurla network task
output = tf.keras.layers.Dense(6, activation='softmax')(x)  # softmax is the activation function for classiciation task

model = tf.keras.Model(inputs=[image_input, object_input], outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights('classify_model.hdf5')

# Predict the label of the test_images
pred = model.predict([image_array, object_array])
pred_final = np.argmax(pred, axis=1)

# Map the label
labels = (test_image.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred_final = [labels[k] for k in pred_final]
result_df['Class'] = pred_final

# 대상 폴더 정의 (이 폴더 내에 클래스별 하위 폴더가 생성됨)
target_folder = 'c:\\Users\\MING\\Desktop\\sdfa\\classified_images\\'
# 이동될 파일 경로의 리스트를 저장할 빈 리스트 생성
destination_paths = []
# creation_time_stamps = []
# date_taken_lst = []
# date_made_lst = []
# 이러한 방식으로 새로운 열을 만들고 초기화할 수 있습니다.
result_df['date_made'] = None  # None으로 초기화하여 모든 값이 NaT(not a time) 상태로 됩니다.
result_df['date_taken'] = None
result_df['moved_image_path'] = None

# 'object' 타입으로 열의 데이터 타입을 명시적으로 설정합니다.
result_df['date_made'] = result_df['date_made'].astype('object')
result_df['date_taken'] = result_df['date_taken'].astype('object')
result_df['moved_image_path'] = result_df['moved_image_path'].astype('object')

for index, row in result_df.iterrows():
    # 사진 찍힌 시간 or 만든 시간 저장
    # 함수를 사용하여 찍은 날짜 가져오기
    result_df.loc[index, 'date_taken'] = get_date_taken(row['result_image_path'])
    result_df.loc[index, 'date_made'] = get_date_made(row['result_image_path'])

    # 클래스별로 폴더 생성
    class_folder = os.path.join(target_folder, row['Class'])
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # 파일의 존재 여부 확인 및 이동
    if os.path.exists(row['result_image_path']):
        # 파일명 추출
        filename = os.path.basename(row['result_image_path'])
        # 최종 대상 경로 계산
        destination_path = os.path.join(class_folder, filename)
        # 이동될 경로 리스트에 추가
        result_df.loc[index, 'moved_image_path'] = destination_path
        # 파일을 클래스 폴더로 이동
        shutil.move(row['result_image_path'], destination_path)
    else:
        print(f"File not found: {row['result_image_path']}")
    result_df.loc[index, result_lst[index].Class_name] = result_lst[index].Count.tolist()

print("Files moved successfully.")

# result_df['moved_image_path']=destination_paths
# result_df['date_taken']=date_taken_lst
# result_df['date_made']=date_made_lst

result_df.to_csv(
    'classified_images/classification_result_{}.csv'.format(datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')))