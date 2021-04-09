import streamlit as st
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import metrics

st.title('How should I spend my advertising dollars?')
st.header('Data Science exploration by Ryan Ward')

st.write("")
st.write("")

st.write("""
        For this exploration, we'll use an advertising dataset made famous by Gareth James, \
        Daniela Witten, Trevor Hastie and Rob Tibshirani in their terrific book - an  \
        Introduction to Statistical Learning.\n
        It contains the local advertising budget (in thousands of dollars) for a company at \
        different locations and the number of sales made (in thousands of units)."""
         )

# Create a text element and let the reader know the data is loading.


# data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
url = "https://www.statlearning.com/s/Advertising.csv"


@st.cache
def load_data(url):
    return pd.read_csv(url, index_col=0)


data = load_data(url)

# Notify the reader that the data was successfully loaded.
# data_load_state.text('Advertising data imported!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Summary statistics')
st.dataframe(data.describe())

st.write("""
        This gives us important information such as:\n
        - the number of rows in the table (200).\n
        - the central tendencies of the means and medians (50%). These are reasonably similar
        which might indicate we are working with non-skewed data.\n
        - The min and max for each column can also be interesting to look into as they may contain outliers.
        """)

st.subheader('It\'s also good to explore some of the relationships between the features (TV, radio, \
              newspaper) and target (sales)')


# Interesting but a bit too complicated

# repeat_chart = alt.Chart(data).mark_circle().encode(
#     alt.X(alt.repeat("column"), type='quantitative'),
#     alt.Y(alt.repeat("row"), type='quantitative')
# ).properties(
#     width=120,
#     height=120
# ).repeat(
#     row=list(data.columns),
#     column=list(data.columns)
# ).interactive()

# st.altair_chart(repeat_chart)


def plot_scatter(x, y):
    st.altair_chart(alt.Chart(data)
                    .mark_circle(size=60)
                    .encode(
                        x=x,
                        y=y,
                        tooltip=['TV', 'radio', 'newspaper', 'sales'])
                    .properties(title=f'{x} vs {y}')
                    )


plot_scatter('TV', 'sales')
plot_scatter('radio', 'sales')
plot_scatter('newspaper', 'sales')

st.subheader('Through visual inspection, it looks like TV and radio show a stronger relationship with sales than \
              newspapers do but we can explore this with machine learning.')

st.subheader('So let\'s create a model!')


st.write('1. First we need to split data into training and testing data. This allows us \
              to evaluate the performance of the model')

test_size = st.slider('Choose the size of your test data: 0.3 represent 30% test size',
                      0.10, 0.40, 0.3)

models = {'Linear Regression (OLS)': LinearRegression(),
          'Random Forest': RandomForestRegressor(),
          'K Neighbours': KNeighborsRegressor(),
          'Support Vector Regression': SVR()
          }


st.write('2. Now let''s choose a model to train.')
model = st.selectbox(
    'What model do you want to try?',
    list(models),
)


y = data['sales']
X = data.drop('sales', axis=1)

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

st.altair_chart(scatter_chart + scatter_chart.transform_regression('Predictions', 'Actuals').mark_line())

mse = metrics.mean_squared_error(evaluation['Actuals'], evaluation['Predictions'])
rmse = round(mse**0.5, 2)

st.write(f'Square Root of the Mean Squared Error (RMSE): {rmse}.  \n \
          This can be interpreted that in general the sales prediction will \
          be out by {int(rmse*1000)} units.')

st.write("""
    Feel free to try other models and see how they perform!.
    """)

st.subheader('Making your own predictions')
st.write("""
    Now for the interesting bit, putting in your own marketing budget and predicting sales.  \n\n \
    Try entering in your budget for TV, radio and newspapers to see what the model predicts.
    """)

TV_value = st.slider('TV budget (in $\'000)', min_value=0, max_value=500, value=0)
radio_value = st.slider('Radio budget (in $\'000)', min_value=0, max_value=500, value=0)
newspaper_value = st.slider('Newspaper budget (in $\'000)', min_value=0, max_value=500, value=0)

if st.button('How many things am I going to sell?'):
    single_prediction = reg.predict([[TV_value, radio_value, newspaper_value]])
    st.write(f'The model predicts you\'ll sell {int(single_prediction[0])}k units a  \
               combined advertising budget of TV: ${TV_value}k, Radio: ${radio_value}k and \
               Newspaper: ${newspaper_value}k.')

    st.write('Feel free to play with the inputs and see if you can find an optimal budget - making \
              the most sales for the least cost!')
    st.subheader('Feel free to reach out to me at ryan@fromlawtodata.com if you have any questions.')
