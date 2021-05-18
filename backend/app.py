from flask import Flask, request
from deepface import DeepFace
import json
# from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS
import os
import dlib
import sqlite3
import shutil
from normal_pic_embedding import normal_pic_embeddings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
from io import BytesIO
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app, supports_credentials=True)

model = DeepFace.build_model('Facenet')

face_point_detector = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

MODEL_DIR = './predict_models'
TEMP_DIR = './temp'
EMBEDDING_DIR = './embeddings'

@app.route('/')
def hello_world():
    return 'Hello World!'


def mutil_disease_predict(face_embedding):
    knn = get_knn()
    result = knn.predict(face_embedding)
    return json.dumps({'resutl': result[0]})

def get_knn():
    embedding_list = os.listdir(EMBEDDING_DIR)
    X = []
    Y = []
    for embeddings_name in embedding_list:
        with open(f'{EMBEDDING_DIR}/{embeddings_name}') as f:
            embeddings = json.load(f)
        X += embeddings
        Y += [embeddings_name.rsplit('.', 1)[0]] * len(embeddings)
    knn = KNeighborsClassifier(n_neighbors=2, weights='distance')
    knn.fit(X, Y)
    return knn

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form.get('disease')
    file = request.files['file']
    img = file.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    dets = face_detector(img, 1)
    if len(dets) != 1:
        return json.dumps({"msg": 'There is not unique face in this picture'})
    det = dets[0]
    face_points = face_point_detector(img, det)
    faces = dlib.full_object_detections()
    faces.append(face_points)
    normed_face = dlib.get_face_chips(img, faces)[0]
    img_np = np.array(normed_face)
    face_embedding = [DeepFace.represent(
        img_np,
        model_name='Facenet',
        model=model,
        enforce_detection=False
    )]

    if disease == 'mutil-disease':
        return mutil_disease_predict(face_embedding)

    pca = joblib.load(f'{MODEL_DIR}/{disease}_pca.pkl')
    clf = joblib.load(f'{MODEL_DIR}/{disease}.pkl')
    face_embedding_pca = pca.transform(face_embedding)
    result = clf.predict(face_embedding_pca)[0]
    # return json.dumps({"files": [result.get_file()], 'resutl': result})
    return json.dumps({'resutl': result})

@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    disease = request.form.get('disease')
    file = request.files['file']
    img = file.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    dets = face_detector(img, 1)
    if len(dets) != 1:
        return json.dumps({"msg": 'There is not unique face in this picture'})
    det = dets[0]
    face_points = face_point_detector(img, det)
    faces = dlib.full_object_detections()
    faces.append(face_points)
    normed_face = dlib.get_face_chips(img, faces)[0]
    normed_face = cv2.resize(normed_face, (96, 112))
    img_np = np.array(normed_face)
    img_np = np.expand_dims(img_np, axis=0)
    print(img_np.shape)
    model = load_model(f'{MODEL_DIR}/{disease}.h5')
    if disease == 'mutil-disease':
        query_sql = "SELECT * FROM predict_nn_model"
        class_list = []
        try:
            conn = sqlite3.connect('./predict_models.db')
            cur = conn.cursor()
            cur.execute(query_sql)
            class_list = cur.fetchall()
        finally:
            cur.close()
            conn.close()
    else:
        class_list = str.split(disease, '-')
    class_list.sort()
    result = class_list[np.argmax(model.predict(img_np))]
    print(result)
    return json.dumps({'resutl': result})

@app.route('/model_list')
def get_model_list():
    query_sql = "SELECT * FROM predict_model"
    model_list = []
    try:
        conn = sqlite3.connect('./predict_models.db')
        cur = conn.cursor()
        cur.execute(query_sql)
        model_list = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    return json.dumps({'model_list': model_list})

@app.route('/nn_model_list')
def get_nn_model_list():
    query_sql = "SELECT * FROM predict_nn_model"
    model_list = []
    try:
        conn = sqlite3.connect('./predict_models.db')
        cur = conn.cursor()
        cur.execute(query_sql)
        model_list = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    return json.dumps({'model_list': model_list})


@app.route('/train', methods=['POST'])
def train():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    file = request.files['file']
    disease_name = file.filename.rsplit(".", maxsplit=1)[0]
    file_stream = BytesIO(file.stream.read())
    shutil._unpack_zipfile(file_stream, TEMP_DIR)
    unpack_file_list = os.listdir(TEMP_DIR)
    # 获取图片实际存在路径
    # 如果压缩包中是一个文件夹且该文件夹中包含所需训练图片，则取该文件夹内的路径为训练图片所在位置
    train_pic_dir = TEMP_DIR
    if len(unpack_file_list) == 1 and os.path.isdir(os.path.join(TEMP_DIR, unpack_file_list[0])):
        train_pic_dir = os.path.join(TEMP_DIR, unpack_file_list[0])
        unpack_file_list = os.listdir(train_pic_dir)
    # 预处理上传图片并直接获取embedding
    X = normal_pic_embeddings[:len(unpack_file_list)]
    Y = ['normal'] * len(X)
    for pic_name in unpack_file_list:
        pic_path = os.path.join(train_pic_dir, pic_name)
        img = cv2.imread(pic_path)
        dets = face_detector(img, 1)
        if len(dets) != 1:
            print(f'There is not unique face in {pic_name}')
            continue
        det = dets[0]
        face_points = face_point_detector(img, det)
        faces = dlib.full_object_detections()
        faces.append(face_points)
        normed_face = dlib.get_face_chips(img, faces)[0]
        img_np = np.array(normed_face)
        X.append(
            DeepFace.represent(img_np, model_name='Facenet',
                               model=model, enforce_detection=False)
        )
        Y.append(disease_name)
    best_score = -1
    best_kernel = ''
    best_pca_dim = -1
    min_dim = len(unpack_file_list)
    min_dim = min(min_dim, 129)
    print(f'searching pca dim in range 1 to {min_dim}')
    for n_dim in range(1, min_dim):
        pca = PCA(n_components=n_dim)
        pca = pca.fit(X)
        X_dr = pca.transform(X)
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        for kernel in kernels:
            clf = SVC(kernel=kernel)
            score = cross_val_score(clf, X_dr, Y, cv=7, scoring='accuracy').mean()
            if score >= best_score:
                best_score = score
                best_pca_dim = n_dim
                best_kernel = kernel
                print("The accuracy under kernel %s and pca dimension %d is %f" % (kernel, n_dim, score))
    # 生成最优pca降维模型
    pca = PCA(n_components=best_pca_dim)
    pca = pca.fit(X)
    X_dr = pca.transform(X)
    joblib.dump(pca, os.path.join(MODEL_DIR, f'{disease_name}-normal_pca.pkl'))
    # 生成最优svm模型
    clf = SVC(kernel=best_kernel, probability=True)
    clf.fit(X_dr, Y)
    joblib.dump(clf, os.path.join(MODEL_DIR, f'{disease_name}-normal.pkl'))

    score = cross_val_score(clf, X_dr, Y, cv=7, scoring='accuracy').mean()

    # 插入数据库中
    insert_sql = 'INSERT INTO predict_model VALUES (?, ?)'
    try:
        conn = sqlite3.connect('./predict_models.db')
        cur = conn.cursor()
        cur.execute(insert_sql, (f'{disease_name}-normal', round(score, 4)))
        conn.commit()
    finally:
        cur.close()
        conn.close()

    with open(os.path.join(EMBEDDING_DIR, f'{disease_name}.json'), 'w') as f:
        json.dump(normal_pic_embeddings[len(unpack_file_list):], f)

    shutil.rmtree(TEMP_DIR)
    return json.dumps({
        'acc': round(score, 4)
    })


if __name__ == '__main__':
    app.run()
