import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Pollution in Graz", page_icon=":office:", layout="wide")


st.title('Pollution in Graz')

image_know = Image.open('Know.jpg')
image_fkit = Image.open('fkit.jpg')

p, d, t,c,pe,s,se,o = st.columns(8)
p.image(image_know,width = 250)
t.image(image_fkit,width = 300)

site = st.sidebar.selectbox(
     'What site would you like to see?',
     ('Nord','Sud','Ost','West','DonBosco'))

algoritam = st.sidebar.selectbox(
     'What algoritam would you like to see',
     ('Neural Network','Random Forest Regression'))

data_to_see = st.sidebar.selectbox(
     'Data',
    ('Local and temporal data',
     'Satelit and temporal data',
     'Local,satelit and temporal data',
     'Local, temporal and traffic data',
     'Satelite, temporal and traffic data',
     'All data'))

left_column, middle_column, right_column,col = st.columns(4)
with left_column:
    st.subheader(f"Site: {site}")

with middle_column:
    st.subheader(f"Algoritam: {algoritam}")

with right_column:
    st.subheader(f"Data: {data_to_see}")


st.markdown("""---""")


def get_data_for_graf(var):
    if var == 'Local and temporal data':
        local = pd.read_csv(r'../data/rezultati_local.csv', index_col=0)
        temp_data = local
        r2_temp = 'r2local'
        rmse_temp ='rmselocal'
        return temp_data,r2_temp,rmse_temp
    elif var == 'Satelit and temporal data':
        satelite = pd.read_csv(r'../data/rezultati_satelit.csv', index_col=0)
        temp_data = satelite
        r2_temp = 'r2satelite'
        rmse_temp = 'rmsesatelite'
        return temp_data,r2_temp,rmse_temp
    elif var == 'Local,satelit and temporal data':
        all_meteo = pd.read_csv(r'../data/rezultati_all_meteo.csv', index_col=0)
        temp_data = all_meteo
        r2_temp = 'r2allmeteo'
        rmse_temp = 'rmseallmeteo'
        return temp_data,r2_temp,rmse_temp
    elif var == 'Local, temporal and traffic data':
        traffic_local = pd.read_csv(r'../data/rezultati_local_traffic.csv', index_col=0)
        temp_data = traffic_local
        r2_temp = 'r2localtraffic'
        rmse_temp = 'rmselocaltraffic'
        return temp_data,r2_temp,rmse_temp
    elif var ==  'Satelite, temporal and traffic data':
        traffic_satelit = pd.read_csv(r'../data/rezultati_satelite_traffic.csv', index_col=0)
        temp_data = traffic_satelit
        r2_temp = 'r2satelitetraffic'
        rmse_temp = 'rmsesatelitetraffic'
        return temp_data,r2_temp,rmse_temp
    elif var == 'All data':
        all_traffic = pd.read_csv(r'../data/rezultati_all_traffic.csv', index_col=0)
        temp_data = all_traffic
        r2_temp = 'r2alltraffic'
        rmse_temp = 'rmsealltraffic'
        return temp_data,r2_temp,rmse_temp

def get_data_from_excel(r2_temp,rmse_temp):
    df_r2 = pd.read_excel(io="../data/resultati.xlsx",engine="openpyxl", sheet_name=r2_temp,index_col=0)
    df_rmse = pd.read_excel(io="../data/resultati.xlsx", engine="openpyxl", sheet_name=rmse_temp,index_col=0)
    return df_r2,df_rmse

def sites(site):
    if site == 'Nord':
        p = 'N_PM10K'
        return p
    if site == 'Ost':
        p = 'O_PM10K'
        return p
    if site == 'West':
        p = 'W_PM10K'
        return p
    if site == 'Sud':
        p = 'S_PM10K'
        return p
    if site == 'DonBosco':
        p = 'D_PM10K'
        return p

def metric(df_r2_t,df_rmse_t,site,algoritam):
    if algoritam == 'Neural Network':
        df_r2_t = df_r2_t.filter(like=site,axis=1)
        df_r2_t = df_r2_t.filter(like='nn_',axis=0)
        df_rmse_t = df_rmse_t.filter(like=site, axis=1)
        df_rmse_t = df_rmse_t.filter(like='nn_', axis=0)
        df_r2_t.index.name = 'Random State'
        df_rmse_t.index.name = 'Random State'
        return df_r2_t,df_rmse_t
    else:
        df_r2_t = df_r2_t.filter(like=site, axis=1)
        df_r2_t = df_r2_t.filter(like='rf_', axis=0)
        df_rmse_t = df_rmse_t.filter(like=site, axis=1)
        df_rmse_t = df_rmse_t.filter(like='rf_', axis=0)
        df_r2_t.index.name = 'Random State'
        df_rmse_t.index.name = 'Random State'
        return df_r2_t,df_rmse_t

def graf_podaci_svi(algo,data,postaja):
    if algo == 'Neural Network':
        temp_postaja = postaja + '_mean_nn_'
        postaja1 = postaja + '_mean'
        test = data[postaja1]
        df = data.filter(like=temp_postaja, axis=1)
        df = pd.concat([df, test], axis=1)
        return df
    else:
        temp_postaja = postaja + '_mean_rf_'
        postaja1 = postaja + '_mean'
        test = data[postaja1]
        df = data.filter(like=temp_postaja, axis=1)
        df = pd.concat([df, test], axis=1)
        return df



site_for_data = sites(site)
df,r2,rmse = get_data_for_graf(data_to_see)
df_r2,df_rmse = get_data_from_excel(r2,rmse)
m2,m3 = metric(df_r2,df_rmse,site,algoritam)

df_graf = graf_podaci_svi(algoritam,df,site_for_data)



fig_line = px.line(df_graf,
                      title=site+' '+algoritam,
                    width=1400,
                   height=500,
                    labels={
                     "value": "Particulate matter",
                     "index": "Date",

                 },
                    template="plotly_white",
                      )



fig_line.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False)))

fig_r2 = px.bar(
    m2,
    x=m2.index,
    y=site,
    title="R2 metric",
    template="plotly_white",)

fig_r2.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False)))

fig_rmse= px.bar(
    m3,
    x=m3.index,
    y=site,
    title="RMSE metric",
    template="plotly_white")

fig_rmse.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False)),)

left_column, right_column = st.columns(2)
right_column.plotly_chart(fig_rmse, use_container_width=True)
left_column.plotly_chart(fig_r2, use_container_width=True)
col1, col2 = st.columns(2)
col1.metric("R2", m2.max().round(4))
col2.metric("RMSE", m3.min().round(4))


st.plotly_chart(fig_line)









hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)