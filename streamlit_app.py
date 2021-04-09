from time import sleep

import streamlit as st
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import metrics

st.write('# How should I spend my advertising dollars?')
st.write('## Data Science exploration by Ryan Ward')

st.write("")
st.write("")

st.write("""
        For this exploration, we'll use an advertising dataset made famous by Gareth James, \
        Daniela Witten, Trevor Hastie and Rob Tibshirani in their terrific book - an  \
        Introduction to Statistical Learning.\n
        It contains the local advertising budget (in $'000) for a company at \
        different locations and the number of sales made of a product (in thousands of units).\n
        We'll start with a quick exploration before conducting a type of machine learning \
        called regression, then we can make some predictions.
        """
         )

url = "https://www.statlearning.com/s/Advertising.csv"


@st.cache
def load_data(url):
    data = pd.read_csv(url, index_col=0)
    data.columns = ['TV', 'Radio', 'Newspaper', 'Sales']
    return data


data = load_data(url)

# Notify the reader that the data was successfully loaded.
# data_load_state.text('Advertising data imported!')

if st.checkbox('Show raw data'):
    st.write('### Raw data')
    st.write(data)

st.write('### Summary statistics')
st.dataframe(data.describe())

st.write("""
        This gives us important information such as:\n
        * **number of rows** in the table (200).\n
        * **central tendencies** of the means and medians (50%). These are reasonably similar
          which might indicate we are working with non-skewed data.\n
        * **min and max** for each column can also be interesting to look into as they may
          contain outliers.
        """)

st.write('It can also be valuable to explore relationships between the features \
          (TV, radio, newspaper) and target (sales)')
st.write("")


def plot_scatter(x, y):
    st.altair_chart(alt.Chart(data)
                    .mark_circle(size=60)
                    .encode(
                        x=x,
                        y=y,
                        tooltip=['TV', 'Radio', 'Newspaper', 'Sales'])
                    .properties(title=f'{x} budget vs {y}')
                    )


plot_scatter('TV', 'Sales')
plot_scatter('Radio', 'Sales')
plot_scatter('Newspaper', 'Sales')

st.write("""
        #### What do the charts tell us?\n
        * **TV budget** increases reliably with sales but plateaus at the higher values\n
        * **Radio budget** also increases with sales but not as strong as TV'\n
        * **Newspaper budget** shows the weakest relationship with sales'\n
        We can explore this further with a type of machine learning called regression.
        """)


st.write('## So let\'s create a model!')

st.write('1. First we need to split data into training and testing data.')
st.write('We\'ll train the model on the training data and validate its performance \
          on the testing data.')

test_size = st.slider('Choose the size of your test data (e.g. 0.30 represent 30% test size)',
                      0.1, 0.4, 0.3)

models = {'Linear Regression (OLS)': LinearRegression(),
          'Random Forest': RandomForestRegressor(),
          'K Neighbours': KNeighborsRegressor(),
          'Support Vector Regression': SVR()
          }

st.write('2. Now let''s choose a model to train.')
st.write('It can be tricky to know which models to try but there is no harm \
          in trying multiple different models and comparing results.')

model = st.selectbox(
    'What model do you want to try?',
    list(models),
)

y = data['Sales']
X = data.drop('Sales', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

reg = models[model]
reg.fit(X_train, y_train)

predict = reg.predict(X_test)

evaluation = pd.DataFrame({
    'Predictions': predict,
    'Actuals': y_test
})

scatter_chart = (alt.Chart(evaluation)
                 .mark_circle(size=60)
                 .encode(
    alt.X("Predictions"),
    alt.Y("Actuals")
)
    .properties(title='Performance of predicted vs actual sales')
)

st.altair_chart(scatter_chart + scatter_chart.transform_regression('Predictions', 'Actuals')
                .mark_line())

mse = metrics.mean_squared_error(evaluation['Actuals'], evaluation['Predictions'])
rmse = round(mse**0.5, 2)

st.write('### How do we know if our model performed well?')
st.write(f'The Square Root of the Mean Squared Error (rmse) is one metric we can use. \
           Here, the rmse was {rmse}.  \n \
           This can be interpreted that in general the sales prediction will be out \
           by {int(rmse*1000)} units.')
st.write('#### Is this good?')
st.write('It depends! For some medical diagnosis model you might need to be far more accurate, but in a \
          business context it might be fine.')

st.write('### Making your own predictions')
st.write("""
          Now for the interesting bit, entering your own marketing budget and predicting sales.\n\n\
          Try entering in your budget for TV, radio and newspapers to see what the model predicts.
         """)

TV_value = st.slider('TV budget (in $\'000)', min_value=0, max_value=500, value=0)
radio_value = st.slider('Radio budget (in $\'000)', min_value=0, max_value=500, value=0)
newspaper_value = st.slider('Newspaper budget (in $\'000)', min_value=0, max_value=500, value=0)

if st.button('How many things am I going to sell?'):
    # st.progress(progress_variable_1_to_100)
    st.spinner()
    with st.spinner(text='Thinking really hard...'):
        sleep(2)
        st.success('Prediction complete')
    single_prediction = reg.predict([[TV_value, radio_value, newspaper_value]])
    st.write(f'The model predicts you\'ll sell {int(single_prediction[0])}k units from a  \
               combined advertising budget of TV: ${TV_value}k, Radio: ${radio_value}k and \
               Newspaper: ${newspaper_value}k.')

    st.write('Tinker with the inputs and see if you can find an optimal budget - making \
              the most sales for the least cost!')
    st.write('')
    st.write('#### Feel free to reach out to me at ryan@fromlawtodata.com if you have \
                   any questions.')
