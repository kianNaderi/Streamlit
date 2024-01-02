import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from surprise import accuracy
from surprise import NMF



rs = 123
#current_directory = os.getcwd()
models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
     return pd.read_csv("ratings.csv")

def load_user_feature():
     return pd.read_csv("user_feature_df.csv")

def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_user_sims():
    return pd.read_csv("sim_users.csv")


def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_course_genre():
    course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
    return pd.read_csv(course_genre_url)

def predict_rating(user_item_matrix, user_similarity, user, item, k=2):
    # Find k-nearest neighbors
    similar_users = np.argsort(user_similarity[int(user)])[::-1][1:k+1]

    # Predict the rating using a weighted average of the neighbors' ratings
    prediction = np.dot(user_similarity[user, similar_users], user_item_matrix.loc[similar_users, item])
    prediction /= np.sum(np.abs(user_similarity[user, similar_users]))

    return prediction

def process_dataset(raw_data):
    
    encoded_data = raw_data.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_id2idx_dict, course_id2idx_dict

def generate_train_test_datasets(dataset, scale=True):
    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])
    

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id

def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

def get_unknown_courses_id(user_id, ratings_df, genre_matrix):
    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_courses = user_ratings['item'].to_list()
    all_courses = set(genre_matrix['COURSE_ID'].values)
    unknown_courses = all_courses.difference(enrolled_courses)
    unknown_course_df = genre_matrix[genre_matrix['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].values
    return unknown_course_ids

def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict 


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def calculate_user_feature_vector(user_id, genre_matrix, ratings_df):
        user_ratings = ratings_df[ratings_df['user'] == user_id]    
        merged_rating_title = pd.merge(genre_matrix, user_ratings, how='left', left_on='COURSE_ID', right_on='item')    
        column_to_multiply = 'rating'    
        columns_to_multiply = merged_rating_title.columns[2:-3]    
        user_vector = merged_rating_title[columns_to_multiply].multiply(merged_rating_title[column_to_multiply], axis=0).sum(skipna=True)
        return user_vector

# Model training
def train(model_name, params):
    if model_name == models[2] or model_name == models[3]:
        ratings_df = load_ratings()
        user_profile_df = load_user_feature()
        genre_matrix = load_course_genre()
        
        features = user_profile_df.loc[:, user_profile_df.columns != 'User']
        user_ids = user_profile_df.loc[:, user_profile_df.columns == 'User']
        user_id = ratings_df['user'].iloc[-1]

        user_feature_vector = calculate_user_feature_vector(user_id, genre_matrix, ratings_df)
        scaler = StandardScaler()
        reshaped_array = user_feature_vector.values.reshape(-1, 1)
        user_feature_vector_scaled = scaler.fit_transform(reshaped_array)

        if model_name == models[3] :
            pca = PCA(n_components=params['pca'])
            features = pca.fit_transform(features)
            user_feature_vector_scaled = pca.transform(user_feature_vector_scaled.reshape(1, -1))

        k_model = KMeans(n_clusters = params['cluster_num'] , random_state=rs)
        k_model.fit(features)
        global user_cluster
        global cluster_df
        user_cluster = k_model.predict(np.array(user_feature_vector_scaled).flatten().reshape(1, -1))
        cluster_labels = k_model.labels_
        cluster_df = combine_cluster_labels(user_ids,cluster_labels)
        
    elif model_name == models[6] or model_name == models[7] or model_name == models[8]:
        ratings_df = load_ratings()
        num_users = len(ratings_df['user'].unique())
        num_items = len(ratings_df['item'].unique())
        class RecommenderNet(keras.Model):
            def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
                """
                Constructor
                :param int num_users: number of users
                :param int num_items: number of items
                :param int embedding_size: the size of embedding vector
                """
                super(RecommenderNet, self).__init__(**kwargs)
                self.num_users = num_users
                self.num_items = num_items
                self.embedding_size = embedding_size
                
                # Define a user_embedding vector
                # Input dimension is the num_users
                # Output dimension is the embedding size
                self.user_embedding_layer = layers.Embedding(
                    input_dim=num_users,
                    output_dim=embedding_size,
                    name='user_embedding_layer',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(1e-6),
                )
                # Define a user bias layer
                self.user_bias = layers.Embedding(
                    input_dim=num_users,
                    output_dim=1,
                    name="user_bias")
                
                # Define an item_embedding vector
                # Input dimension is the num_items
                # Output dimension is the embedding size
                self.item_embedding_layer = layers.Embedding(
                    input_dim=num_items,
                    output_dim=embedding_size,
                    name='item_embedding_layer',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(1e-6),
                )
                # Define an item bias layer
                self.item_bias = layers.Embedding(
                    input_dim=num_items,
                    output_dim=1,
                    name="item_bias")
                
            def call(self, inputs):
                """
                method to be called during model fitting
                
                :param inputs: user and item one-hot vectors
                """
                # Compute the user embedding vector
                global user_vector
                global item_vector
                user_vector = self.user_embedding_layer(inputs[:, 0])
                user_bias = self.user_bias(inputs[:, 0])
                item_vector = self.item_embedding_layer(inputs[:, 1])
                item_bias = self.item_bias(inputs[:, 1])
                dot_user_item = tf.tensordot(user_vector, item_vector, 2)
                # Add all the components (including bias)
                x = dot_user_item + user_bias + item_bias
                # Sigmoid output layer to output the probability
                return tf.nn.relu(x)
        
        global user_id2idx_dict , course_id2idx_dict
        encoded_data, user_id2idx_dict, course_id2idx_dict = process_dataset(ratings_df)
        x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_data)
        embedding_size = params['embedding_size']
        global nn
        nn = RecommenderNet(num_users, num_items, embedding_size)
        nn.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
        history = nn.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs= params['epoch_size'])
        
        
        if model_name == models[7] or model_name == models[8]:
            from sklearn.preprocessing import LabelEncoder
            global user_feature , item_feature
            genre_matrix = load_course_genre()
            all_courses = set(ratings_df['item'].values)
            user_id = ratings_df['user'].iloc[-1]
            unselected_course_ids = get_unknown_courses_id(user_id, ratings_df, genre_matrix)
            
            user_latent_features = nn.get_layer('user_embedding_layer').get_weights()[0]
            item_latent_features = nn.get_layer('item_embedding_layer').get_weights()[0]
            
            user_emb_rate = pd.Series(ratings_df['user'].unique())
            item_emb_rate = pd.Series(ratings_df['item'].unique())
            
            user_emb = pd.DataFrame(user_latent_features)
            user_emb = pd.concat([user_emb, user_emb_rate], axis=1)
            user_emb.columns = [f"UFeature{num}" for num in range(embedding_size)] + ['user']
            user_feature = user_emb.iloc[-1]

            item_emb = pd.DataFrame(item_latent_features , columns= [f"feature{num}" for num in range(embedding_size)])
            item_emb = pd.concat([item_emb, item_emb_rate], axis=1)
            item_emb.columns = [f"CFeature{num}" for num in range(embedding_size)] + ['item']
            item_feature = item_emb[(item_emb['item'].isin(unselected_course_ids)) & (item_emb['item'].isin(all_courses))]

            user_emb_merged = pd.merge(ratings_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
            merged_df = pd.merge(user_emb_merged, item_emb, how='left', left_on='item', right_on='item').fillna(0)
            
            u_feautres = [f"UFeature{i}" for i in range(embedding_size)]
            c_features = [f"CFeature{i}" for i in range(embedding_size)]

            user_embeddings = merged_df[u_feautres]
            course_embeddings = merged_df[c_features]
            ratings = merged_df['rating']

            regression_dataset = user_embeddings + course_embeddings.values
            regression_dataset.columns = [f"Feature{i}" for i in range(embedding_size)]
            regression_dataset['rating'] = ratings

            X = regression_dataset.iloc[:, :-1]
            y_raw = regression_dataset.iloc[:, -1]

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.values.ravel())
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=.2 , random_state=rs)

            
            global model

            if model_name == models[7] :
                from sklearn import linear_model
                model = linear_model.Ridge(alpha = 0.2)
                model.fit(X_train , y_train)
            if model_name == models[8] :
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(penalty='l2')
                model.fit(X_train , y_train)
                
            
            

# Prediction
def predict(model_name, user_ids, params):
    # ratings_df = load_ratings()
    # all_users = ratings_df['user'].unique()
    # sample_users = np.random.choice(all_users, 200, replace=False)

    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
# Course Similarity model
        if model_name == models[0]:
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
# User Profile model
        elif model_name == models[1]:
            genre_matrix = load_course_genre()
            ratings_df = load_ratings()
            
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            merged_rating_title =pd.merge(genre_matrix ,user_ratings , how='left' , left_on='COURSE_ID' , right_on= 'item')
            # Specify the column to multiply with other columns
            column_to_multiply = 'rating'
            columns_to_multiply = merged_rating_title.columns[2:-3]
            user_vector = merged_rating_title[columns_to_multiply] = merged_rating_title[columns_to_multiply].multiply(merged_rating_title[column_to_multiply], axis=0).sum(skipna=True)

            # enrolled_courses = ratings_df[ratings_df['user'] == user_id]['item'].to_list()
            enrolled_courses = user_ratings['item'].to_list()
            all_courses = set(genre_matrix['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_courses)
            unknown_course_df = genre_matrix[genre_matrix['COURSE_ID'].isin(unknown_courses)]
            unknown_course_ids = unknown_course_df['COURSE_ID'].values
            # user np.dot() to get the recommendation scores for each course
            course_matrix = unknown_course_df.iloc[:, 2:].values
            recommendation_scores = np.dot(course_matrix, user_vector)
            recommendation_scores = (recommendation_scores / np.max(recommendation_scores))
            # Append the results into the users, courses, and scores list
            for i in range(0, len(unknown_course_ids)):
                score = recommendation_scores[i]
                # Only keep the courses with high recommendation score
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(unknown_course_ids[i])
                    scores.append(recommendation_scores[i])
                else :
                    continue
        
# Clustering model
        elif model_name == models[2] or model_name == models[3]: 
            ratings_df = load_ratings()[['user', 'item']]
            test_users_labelled = pd.merge(ratings_df, cluster_df, left_on='user', right_on='user')
            courses_cluster = test_users_labelled[['item', 'cluster']]
            courses_cluster['count'] = [1] * len(courses_cluster)
            popular_courses = courses_cluster.groupby(['cluster','item']).agg(enrollments = ('count','sum')).reset_index().sort_values(by='enrollments' , ascending = False)


            user_subset = test_users_labelled[test_users_labelled['user'].astype(float) == user_id]  
            cluster_courses = set(popular_courses[popular_courses['cluster'] == user_cluster[0]]['item'])
            user_items = user_subset['item']
            unseen_courses = cluster_courses.difference(user_items)
            popular_courses_cluster = popular_courses[(popular_courses['cluster']==user_cluster[0]) & (popular_courses['enrollments'] > params['min_enroll'])]['item']
            mask = popular_courses_cluster.isin(unseen_courses)
            courses = list(popular_courses_cluster[mask])
            users = [user_id] * len(courses)
            scores = [1] * len(courses)
# KNN model

        elif model_name == models[4] : 
            
            ratings_df = load_ratings()
            genre_matrix = load_course_genre()

            reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
            coruse_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
            trainset, testset = train_test_split(coruse_dataset, test_size=.3)
            algo = KNNBasic(k= params['num_neighbors'], sim_option = {'name': 'MSD', 'user_based': True})

            

            algo.fit(trainset)

            unselected_course_ids = get_unknown_courses_id(user_id, ratings_df, genre_matrix)
            print(unselected_course_ids)

            for item in unselected_course_ids:
                predictions  = algo.predict(str(user_id), item)
                print(params['score_threshold'])
                print(predictions.est)
                if predictions.est == 3:
                    users.append(user_id)
                    courses.append(item)
                    scores.append(predictions.est)
# NMF model
        elif model_name == models[5] :
            ratings_df = load_ratings()
            genre_matrix = load_course_genre()
            unselected_course_ids = get_unknown_courses_id(user_id, ratings_df, genre_matrix)
            reader = Reader(
                line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
            course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
            trainset, testset = train_test_split(course_dataset, test_size=.2)
            algo = NMF(n_factors = params['num_factors'])
            algo.fit(trainset)



            for item in unselected_course_ids:
                prediction = algo.predict(str(user_id), item)
                if prediction.est == 3:
                    users.append(int(user_id))
                    courses.append(item)
                    scores.append(prediction.est)

# NN model        
        elif model_name == models[6] :
            ratings_df = load_ratings()
            genre_matrix = load_course_genre()
            all_courses = set(ratings_df['item'].values)
            unselected_course_ids = get_unknown_courses_id(user_id, ratings_df, genre_matrix)
            a = user_id2idx_dict[user_id] 
            for item in unselected_course_ids:
                if item in all_courses:     
                    b = course_id2idx_dict[item]
                    prediction = nn.predict([[a,b]])
                    if prediction >=  params['min_threshold'] / 100:
                        users.append(user_id)
                        courses.append(item)
                        scores.append(prediction[0][0])
                else :
                    continue

# Regression and Classification model        
        elif model_name == models[7] or model_name == models[8]:
            for index, row in item_feature.iterrows():
                item = row['item']
                aggrigated = row[:-1].values + user_feature[:-1]
                # aggrigated = item_feature.loc[].values[0][:-1] + user_feature[:-1]
                prediction = model.predict(np.array(aggrigated).reshape(1, -1))
                print(prediction)
                if prediction >=  params['min_threshold'] / 100:
                    users.append(user_id)
                    courses.append(item)
                    scores.append(prediction[0])
                else :
                    continue
    
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE']).sort_values(by='SCORE' ,ascending=False)
    return res_df


