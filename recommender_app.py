import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# streamlit run recommender_app.py
# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()

@st.cache_data
def load_course_genre():
    return backend.load_course_genre()

# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        course_genre_df = load_course_genre()
    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):    
    # Course similarity model
    if model_name in backend.models:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name , params)
        st.success('Done!')
    # User profile model



def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, user_ids, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['sim_threshold'] = course_sim_threshold
# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('Course Score Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
    params['sim_threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2] :
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    params['cluster_num'] = cluster_no
    min_enroll = st.sidebar.slider('Minimum Enrollments',
                                   min_value=20, max_value=100,
                                   value=20, step=5)
    params['min_enroll'] = cluster_no
elif model_selection == backend.models[3]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    params['cluster_num'] = cluster_no
    min_enroll = st.sidebar.slider('Minimum Enrollments',
                                   min_value=20, max_value=100,
                                   value=20, step=5)
    params['min_enroll'] = cluster_no
    pca = st.sidebar.slider('Principal Component Analysis',
                                   min_value=1, max_value=14,
                                   value=12, step=1)
    params['pca'] = pca
elif model_selection == backend.models[4]:
    num_neighbors = st.sidebar.slider('Number of Neighbors',
                                              min_value=0, max_value=40,
                                              value=10, step=1)
    params['num_neighbors'] = num_neighbors
    course_score_threshold = st.sidebar.slider('Course Score Threshold ',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['score_threshold'] = course_score_threshold

elif model_selection == backend.models[5]:
    num_factors = st.sidebar.slider('Number of Factors',
                                              min_value=0, max_value=20,
                                              value=15, step=1)
    params['num_factors'] = num_factors

elif model_selection == backend.models[6] or model_selection == backend.models[7] or model_selection == backend.models[8]:
    embedding_size = st.sidebar.slider('Embedding Size',
                                              min_value=0, max_value=20,
                                              value=15, step=1)
    params['embedding_size'] = embedding_size
    epoch_size = st.sidebar.slider('Epoch Size',
                                              min_value=0, max_value=10,
                                              value=2, step=1)
    params['epoch_size'] = epoch_size
    min_threshold = st.sidebar.slider('Min Score Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    
    params['min_threshold'] = min_threshold

if 'new_id' not in st.session_state:
    st.session_state.new_id = None

# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    if selected_courses_df.shape[0] > 0:
        st.session_state.new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
        train(model_selection, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button :
    print(st.session_state.new_id)
    if st.session_state.new_id is not None:
    # Create a new id for current user session
    # new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
        user_ids = [st.session_state.new_id]
        res_df = predict(model_selection, user_ids, params)
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        st.table(res_df)
    else:
        st.write("Please click on 'Train Model' before recommending new courses.")
