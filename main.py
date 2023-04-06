import torch
from model import Generator
from data import Dataset
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import streamlit as st
from io import StringIO
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

def load_model():
    model = Generator(225, 225)

    state_dict = torch.load('./net_state_dict.pt')
    model.load_state_dict(state_dict)
    return model
model = load_model()

def inference(data):
    model.eval()
    results = []
    with torch.no_grad():
        for x in data:
            value = model(x.view(-1, len(x))).squeeze()
            results.append(value)
    return results

def convert_csv(content):
    with open('csv.csv', 'a+') as file:
        for value in content:
            file.write(','.join([str(item) for item in value.tolist()])+'\n')

st.title('BCG to ECG signal')
uploaded_file = st.file_uploader("Upload BCG csv file")
if uploaded_file is not None:

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string = stringio.read()
    data = Dataset(string)
    results = inference(data)

    convert_csv(results)

    with open('csv.csv') as csv:
        st.download_button(
            label="Download ECG data as CSV",
            data=csv,
            file_name='ecg.csv',
            mime='text/csv',
        )
    os.remove('csv.csv')

    st.write('选择需要可视化的数据')
    num_data = st.number_input('选择需要可视化的数据', min_value=1, max_value=len(results), step=1, format='%d')
    if st.button('生成可视化结果'):
        fig = plt.figure()
        plt.plot(data[num_data], label = 'bcg signal')
        plt.plot(results[num_data], label = 'ecg signal')
        plt.grid()
        plt.legend()
        st.pyplot(fig)