import json
import sqlite3
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import train_test_split

pretrained_model_list = {
    'apert-normal': 1.0,
    'angelman-normal': 1.0,
}

pretrained_nn_model_list = {
    'apert-normal': 1.0,
    'Fragile_X-normal': 0.75,
}


def init_db(cur, conn):
    create_table_sql = "CREATE TABLE predict_model (model_name TEXT, test_accuracy REAL);"
    cur.execute(create_table_sql)
    conn.commit()



def insert_pretrained_model(cur, conn):
    insert_model_sql = "INSERT INTO predict_model VALUES (?, ?);"
    data = [(key, value) for key, value in pretrained_model_list.items()]
    cur.executemany(insert_model_sql, data)
    conn.commit()


def query_all_models(cur):
    query_sql = "SELECT * from predict_model;"
    cur.execute(query_sql)
    model_list = cur.fetchall()
    print(model_list)
    query_sql = "SELECT * from predict_nn_model;"
    cur.execute(query_sql)
    model_list = cur.fetchall()
    print(model_list)

def update_model(cur, conn, model_name, acc):
    update_sql = "UPDATE predict_model SET test_accuracy=? WHERE model_name=?;"
    cur.execute(update_sql, (acc, model_name))
    conn.commit()

def delete_model(cur, conn, model_name):
    delete_sql = "DELETE FROM predict_model WHERE model_name=?;"
    cur.execute(delete_sql, [model_name])
    conn.commit()

def init_mutil_disease(cur, conn):
    EMBEDDING_DIR = '../embeddings'
    embedding_list = os.listdir(EMBEDDING_DIR)
    X = []
    Y = []
    for embeddings_name in embedding_list:
        with open(os.path.join(EMBEDDING_DIR, embeddings_name)) as f:
            embeddings = json.load(f)
        X += embeddings
        Y += [embeddings_name.rsplit('.', 1)[0]] * len(embeddings)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=2, weights='distance')
    knn.fit(Xtrain, Ytrain)
    score = knn.score(Xtest, Ytest)
    insert_model_sql = "INSERT INTO predict_model VALUES (?, ?);"
    cur.execute(insert_model_sql, ('mutil-disease', round(score, 4)))
    conn.commit()

def init_nn_model(cur, conn):
    create_table_sql = "CREATE TABLE predict_nn_model (model_name TEXT, test_accuracy REAL);"
    cur.execute(create_table_sql)
    insert_model_sql = "INSERT INTO predict_nn_model VALUES (?, ?);"
    data = [(key, value) for key, value in pretrained_nn_model_list.items()]
    cur.executemany(insert_model_sql, data)
    conn.commit()

def insert_nn_model(cur, conn, data):
    insert_model_sql = "INSERT INTO predict_nn_model VALUES (?, ?);"
    cur.execute(insert_model_sql, data)
    conn.commit()

if __name__ == '__main__':
    is_first_time = not os.path.exists('../predict_models.db')
    conn = sqlite3.connect('../predict_models.db')
    cur = conn.cursor()
    if is_first_time:
        print('init db')
        init_db(cur, conn)
        insert_pretrained_model(cur, conn)
        init_nn_model(cur, conn)
    else:
        query_all_models(cur)
    # init_nn_model(cur, conn)
    # insert_nn_model(cur, conn, ('mutil-disease', 0.8984))
    cur.close()
    conn.close()
    # update_model(cur, conn, 'angelman-normal', 0.9008)
    # update_model(cur, conn, 'apert-normal', 0.9259)
    # delete_model(cur, conn, 'Progeria-normal')
    # init_mutil_disease(cur, conn)
    # delete_model(cur, conn, 'mutil-disease')
    # cur.close()
    # conn.close()


