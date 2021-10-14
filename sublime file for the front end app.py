import streamlit as st
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modelTrainin=st.beta_container()











@st.cache
def get_data(filename):
	taxi_data=pd.read_csv(filename)
	return taxi_data

with header:
	st.title("Welcome to this page you are mine")
	st.text("In this project i write transcivtion of taxi NYC")
	



with dataset:
	st.header('NYC taxi dataset')
	st.text("In this project i use the dataset that i found in www.nyctaxi.com")

	taxi_data=get_data("data/NYC taxi.csv")
	st.write(taxi_data.head())
	
	st.subheader("Pick up location ID distribution on NYC taxi")
	pulocation_dist=pd.DataFrame(taxi_data["PULocationID"].value_counts())
	st.bar_chart(pulocation_dist)



with features:
	st.header("The features that i careated ")

	st.markdown("* **First feature:** I careated this feature because of this....I calucuate it using ing this logic....")

	st.markdown("* **Second feature:** 2nd feature,because of this....I calucuate it using ing this logic....")

with modelTrainin:
	st.header("Time to train the model")
	st.text("**Here you get to choose this hyperparamaters of the model and see how the proformace changes!**")

	sel_col,disp_col=st.beta_columns(2)

	max_depth=sel_col.slider("What shoul max_depth of the model?",min_value=10,max_value=100,value=20,step=10)

	n_estimators=sel_col.selectbox("How many trees should there?", options=[100,200,300,400,500,600,'No limit'],index=0)

	sel_col.text("Here is a list of features in y data:")
	sel_col.write(taxi_data.columns)
	input_feature=sel_col.text_input("Which feature should be used as input feature?",'PULocationID')
	if n_estimators=='No limit':
		regr=RandomForestRegressor(max_depth=max_depth)
	else:
		regr=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

	x=taxi_data[[input_feature]]
	y=taxi_data[["trip_distance"]]

	regr.fit(x,y)
	prediction=regr.predict(y)

	disp_col.subheader("Mean absolute error of the model is:")
	disp_col.write(mean_absolute_error(y,prediction))
	disp_col.subheader("Mean squard error of the model is:")
	disp_col.write(mean_squared_error(y,prediction))
	disp_col.subheader("R squard score of the model is:")
	disp_col.write(r2_score(y,prediction))


if st.checkbox("LIKE it or LOVE it"):  
   st.text("ThankYou from @Saurabh")

st.text("Thank You @ Misra_Turp(youtube) for this project and \n  your videos helped me to start with STREAMLIT ")