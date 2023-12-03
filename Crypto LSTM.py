import streamlit as st
from datetime import date
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('BTC-USD', 'ETH-USD')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


data = data.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#st.write(scaled_data)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
       st.write(x_train)
       st.write(y_train)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#x_train.shape


#model = Sequential()
#model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#model.add(LSTM(50, return_sequences = False))
#model.add(Dense(25))
#model.add(Dense(1))
#model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
#model.fit(x_train, y_train, batch_size = 1, epochs = 15)

model=Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
#model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 15)


model.save('ModelReviews.h5')

# model evaluation
from keras.models import load_model

model = load_model('ModelReviews.h5')
scores = model.evaluate(x_train, y_train)

#LSTM_accuracy = scores[1]*100

#print('Test accuracy: ', scores[1]*1000, '%')
#ACC=(scores[1]*1000)
#st.write(ACC)

test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range (60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rsme = np.sqrt(np.mean(predictions - y_test) ** 2)
st.write("RSME Score is below:")
st.write(rsme)
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.write("Click and drag on the plot to zoom in, you can reset using the top right option")

fig = go.Figure()
fig.add_trace(go.Scatter(x = train.index, y = train.Close,
                    mode='lines',
                    name='Close',
                    marker_color = '#1F77B4'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Close,
                    mode='lines',
                    name='Val',
                    marker_color = '#FF7F0E'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Predictions,
                    mode='lines',
                    name='Predictions',
                    marker_color = '#2CA02C'))

fig.update_layout(
    title='Model',
    titlefont_size = 28,
    hovermode = 'x',
    xaxis = dict(
        title='Date',
        titlefont_size=16,
        tickfont_size=14),
    
    height = 800,
    
    yaxis=dict(
        title='Close price in USD ($)',
        titlefont_size=16,
        tickfont_size=14),
    legend=dict(
        y=0,
        x=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'))

st.plotly_chart(fig)
