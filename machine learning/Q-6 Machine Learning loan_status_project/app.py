from flask import Flask,render_template,request,Response
import pickle
import pandas as pd
from  flask_cors import CORS,cross_origin
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)
CORS(app)

@app.route('/',methods=["GET"])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
@cross_origin()
def predict():
    try:
       inputs= request.form
       print(inputs)
       l = []
       for i in inputs.values():
           l.append(i)
       print(l)
       df=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')
       prediction_df = pd.DataFrame(l, index=df.drop(columns=['Loan_Status']).columns).T
       print(prediction_df)
       if not os.path.isdir('Output/'):
           os.makedirs('Output/',exist_ok=True)
       prediction_df.to_csv("Output/Output.csv",header=True,index=False)
       df_processed=pd.read_csv('preprocessed.csv').drop(columns=['Loan_Status'])
       df=pd.read_csv("Output/Output.csv")
       print(df.shape)
       df.drop(columns=["Loan_ID"], inplace=True)
       df_final = pd.get_dummies(df, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], dtype=np.int64)
       for col in df_processed:
           if col not in df_final:
               df_final[col] = 0
       print(df_final)
       df_ordered = df_final[df_processed.columns]
       #df_ordered.to_csv("input1.csv",header=True,index=False)
       #scaler = StandardScaler()
       #scaled_feature = pd.DataFrame(scaler.fit_transform(df_ordered), columns=df_ordered.columns)
       #scaled_feature.to_csv("scaled_input.csv",header=True,index=False)
       #pred=df_ordered.to_numpy().reshape(1,-1)

       model = pickle.load(open('gradient_boosting_loan_status_prediction.pkl', 'rb'))
       model.predict(df_ordered)
       result=model.predict(df_ordered)

       result_dict={0:'N',1:'Y'}


       return render_template('index.html', result_text="Loan Status is {}".format(result_dict[result[0]]))

    except Exception as e:
        print("exception ocurred %s" %e)



if __name__=="__main__":
    app.run()
