from joblib import load
import streamlit as st
 
# loading the trained model
pickle_in = open('D:/Dilna/5_Machine_Learning/streamlitdeploy/model.pkl', 'rb') 
classifier = load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Age,NumTimesPrg, PlGlcConc, BloodP, SkinThick, TwoHourSerIns, BMI, DiPedFunc):   
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Age,NumTimesPrg, PlGlcConc, BloodP, SkinThick, TwoHourSerIns, BMI, DiPedFunc]])
     
    if prediction == 0:
        pred = 'No'
    else:
        pred = 'Yes'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Diabetes Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # following lines create boxes in which user can enter data required to make prediction
    Age = st.number_input("Enter Age: ")
    NumTimesPrg = st.number_input("Number of times pregnant: ") 
    PlGlcConc = st.number_input("Plasma glucose concentration: ")
    BloodP = st.number_input("Diastolic blood pressure: ") 
    SkinThick = st.number_input("Triceps skin fold thickness: ")
    TwoHourSerIns = st.number_input("2-Hour serum insulin: ")
    BMI = st.number_input("Body mass index (BMI): " )
    DiPedFunc = st.number_input("Diabetes pedigree function: ")
    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Age,NumTimesPrg, PlGlcConc, BloodP, SkinThick, TwoHourSerIns, BMI, DiPedFunc) 
        st.success('Are you diabetic? {}'.format(result))
        #print(LoanAmount)
     
if __name__=='__main__': 
    main()
