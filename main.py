import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import gradio as gr


df2 = pd.read_csv("data/heart.csv")


train, valid, test = np.split(df2.sample(frac=1), [int(0.6*len(df2)), int(0.8*len(df2))])
# i didn't use the validation set yet just made them for future use

# pre proccesing
x_train = pd.DataFrame(train).drop('output',axis=1)
y_train = pd.DataFrame(train)['output']

x_valid = pd.DataFrame(valid).drop('output',axis=1)
y_valid = pd.DataFrame(valid)['output']

x_test = pd.DataFrame(test).drop('output',axis=1)
y_test = pd.DataFrame(test)['output']



# gaussian probability model
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train,y_train)

y_pred = nb_model.predict(x_test)

print(classification_report(y_test,y_pred))


lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train,y_train)

y_pred = lg_model.predict(x_test)

print(classification_report(y_test,y_pred))

#function used for gradio passes user input and typecasts the gradio objects into integers or floats, and creates a single instance data frame for prediction
def dinput(age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall):
        #mapping the user input into integers
        restecg_mapping = {
        "Value 0: normal": 0,
        "Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": 1,
        "Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria": 2
    }
    
        cp_mapping = {
        "Value 1: typical angina": 0,
        "Value 2: atypical angina": 1,
        "Value 3: non-anginal pain": 2,
        "Value 4: asymptomatic": 3
    }
        gender_mapping = {
        "Male": 0,
        "Female": 1
    }       
        #boolean type casting 0 1
        exng = int(bool(exng))
        fbs = int(bool(fbs))

        #type casting the data into a dict, then transforming the dict to a data frame
        data = {
        'age':[int(age)],
        'sex':[gender_mapping.get(sex,0)],
        'cp':[cp_mapping.get(cp,0)],
        'trtbps':[int(trtbps)],
        'chol':[int(chol)],
        'fbs':[fbs],
        'restecg':[restecg_mapping.get(restecg,0)],
        'thalachh':[int(thalachh)],
        'exng':[exng],
        'oldpeak':[int(oldpeak)],
        'slp':[float(slp)],
        'caa':[int(caa)],
        'thall':[int(thall)]

    }
        

        df3 = pd.DataFrame(data)
        print(df3)

        #prediction
        y_pred = lg_model.predict(df3)
        y_pred2 = nb_model.predict(df3)
        return y_pred, y_pred2









# gradio web platform
with gr.Blocks() as demo:
    with gr.Row():
        age = gr.Slider(minimum=18,maximum=100,step=1)
        sex = gr.Dropdown(choices=["Male","Female"],label="gender?")

        cp = gr.Dropdown(choices=["Value 1: typical angina","Value 2: atypical angina","Value 3: non-anginal pain","Value 4: asymptomatic"],label="type of angina")
        
        trtbps = gr.Slider(minimum=100,maximum=200,label="resting heart rate?",step=1)
        chol = gr.Slider(minimum=150,maximum=350,label="cholerstrol level",interactive=True,step=1)
        fbs = gr.Checkbox(label="is the paitient fasting sugar level >120mg ?")
        restecg = gr.Dropdown(choices=["Value 0: normal","Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)","Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria"],label="resting electrocardiographic results?")
        thalachh = gr.Slider(maximum=200,minimum=100,label="maximum heart rate achieved?",step=1,)
        exng = gr.Checkbox(label="does the paitent have exercise induced angina")
        oldpeak = gr.Slider(minimum=0,maximum=6.2, step=0.1,label="previous peak")
        slp = gr.Slider(minimum=0,maximum=2,label="Slope",step=1)
        caa = gr.Slider(minimum=0,maximum=4,label="number of Major vessels",step=1)
        thall = gr.Slider(minimum=0,maximum=3,label="thall rate",step=1)
    with gr.Column():
        run = gr.Button("Run prediction")
        output1 = gr.Textbox(label="logistic regression prediction")
        output2 = gr.Textbox(label=" Gaussian model classification")

    event = run.click(
        fn=dinput,
        inputs=[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall],
        outputs=[output1,output2]
    )


demo.launch(share=False)





        
    
    






