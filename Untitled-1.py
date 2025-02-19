# %%
from pycaret.classification import *
import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import random

# %%
# 固定随机种子
np.random.seed(42)
random.seed(42)

# %%

#加载模型
loaded_model = load_model(r'Ada Boost Classifier') #不需要后缀
#loaded_model
#获取模型的pipeline
pipeline = loaded_model[:-1]

# %%
#对采集数据进行字典匹配转换为01 后期进入模型

#Halo_sign absent=0，Exists=1
Halo_sign_dict = {'absent':0, 'Exists':1}
#Gender Male=0 Female=1
Gender_dict = {"Male":0,"Female":1}
#Composition Others=0  Solid=1
Composition_dict = {'Others':0, 'Solid':1}
#Shape Others=0 Microlobulated=1
Shape_dict = {'Others':0, 'Microlobulated':1}
#Echogenicity Other=0 Hypoechogenicity=1
Echogenicity_dict = {'Others':0, 'Hypoechogenicity':1}
#Echogenic_foci Other=0 Microcalcification=1
Echogenic_foci_dict = {'Others':0, 'Microcalcification':1}
#Margin Smooth=0 Irregular=1
Margin_dict = {'Smooth':0, 'Irregular':1}
#ATR Wider_than_tall=0  Taller_than_wide=1
ATR_dict = {'Wider_than_tall':0, 'Taller_than_wide':1}
#Pathological_diagnosis 0 Benign 1 Malignant
Pathological_diagnosis_dict = {0:'Benign', 1:'Malignant'}

# %%
#读取excel文件
import pandas as pd
data = pd.read_excel(r'3-0-training_data_use.xlsx')
data_shap_input = pipeline.transform(data)
#删除 data_shap_input 中的Pathological_diagnosis列
data_shap_input = data_shap_input.drop(columns='Pathological_diagnosis')


# %%
#构建信息输入界面

#性别
Gender= st.selectbox(
    label = 'Please select your gender',
    options = ('Male', 'Female'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#年龄
Age = st.number_input(
    label = 'Please input your age(years):',
    # min_value = 15.4,
    # max_value = 83,
    value = 30,
    #step = 1,
    format = '%d',
    #help = 'Please input your age in years'
)
#BMI 
BMI = st.number_input(
    label = 'Please input your BMI(Kg/m^2):',
    # min_value = 0,
    # max_value = 100,
    value = 15.6,
    step = 0.1,
    format = '%.1f',
    #help = 'Please input your BMI in Kg/m^2'
)
#Halo_sign  absent/Exists
Halo_sign = st.selectbox(
    label = 'Please select your Halo_sign',
    options = ('absent', 'Exists'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Posterior_echo Absent_of_shadowing/Shadowing/Posterior_attenuation
Posterior_echo = st.selectbox(
    label = 'Please select your Posterior_echo',
    options = ('Absent_of_shadowing', 'Shadowing', 'Posterior_attenuation'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Intra_BFS  Absent/Less/Rich
Intra_BFS = st.selectbox(
    label = 'Please select your Intra_BFS',
    options = ('Absent', 'Less', 'Rich'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Peri_BFS Absent/Less/Rich
Peri_BFS = st.selectbox(
    label = 'Please select your Peri_BFS',
    options = ('Absent', 'Less', 'Rich'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Location Right_lobe/Left_lobe/Isthmus
Location = st.selectbox(
    label = 'Please select your Location',
    options = ('Right_lobe', 'Left_lobe', 'Isthmus'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Maximum_diameter
Maximum_diameter = st.number_input(
    label = 'Please input your Maximum_diameter(mm):',
    # min_value = 0,
    # max_value = 100,
    value = 0.5,
    step = 0.1,
    format = '%.1f',
    help = 'Please input your Maximum_diameter in mm'
)
#Composition Others/Solid
Composition = st.selectbox(
    label = 'Please select your Composition',
    options = ('Others', 'Solid'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Shape  Microlobulated/Others
Shape = st.selectbox(
    label = 'Please select your Shape',
    options = ('Microlobulated', 'Others'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Echogenicity Others/Hypoechogenicity
Echogenicity = st.selectbox(
    label = 'Please select your Echogenicity',
    options = ('Others', 'Hypoechogenicity'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Echogenic_foci Others/Microcalcification
Echogenic_foci = st.selectbox(
    label = 'Please select your Echogenic_foci',
    options = ('Others', 'Microcalcification'),
    index = 1,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#Margin Smooth/Irregular
Margin = st.selectbox(
    label = 'Please select your Margin',
    options = ('Smooth', 'Irregular'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)
#ATR Wider_than_tall/Taller_than_wide
ATR = st.selectbox(
    label = 'Please select your ATR',
    options = ('Wider_than_tall', 'Taller_than_wide'),
    index = 0,
    format_func = str,
    #help = '如果您不想透露，可以选择保密'
)



# %%
#对采集数据进行转换处理 生成predict_data数据框
#顺序 BMI	Halo_sign	Posterior_echo	Intra_BFS	Peri_BFS	Age	
#Gender	Location	Maximum_diameter	Composition	Shape	Echogenicity	Echogenic_foci	Margin	ATR

import pandas as pd
predict_data = pd.DataFrame({
    'BMI':[BMI],
    'Halo_sign':[Halo_sign_dict[Halo_sign]],
    'Posterior_echo':[Posterior_echo],
    'Intra_BFS':[Intra_BFS],
    'Peri_BFS':[Peri_BFS],
    'Age':[Age],
    'Gender':[Gender_dict[Gender]],
    'Location':[Location],
    'Maximum_diameter':[Maximum_diameter],
    'Composition':[Composition_dict[Composition]],
    'Shape':[Shape_dict[Shape]],
    'Echogenicity':[Echogenicity_dict[Echogenicity]],
    'Echogenic_foci':[Echogenic_foci_dict[Echogenic_foci]],
    'Margin':[Margin_dict[Margin]],
    'ATR':[ATR_dict[ATR]]
})

predict_data_new = pipeline.transform(predict_data)

# %%
#把 predict_data 横向拼接到 data_shap_input
data_shap_input_new = pd.concat([data_shap_input,predict_data_new],axis=0)
#data_shap_input_new



# %%
#预测
if st.button("Predict"):
    predition = predict_model(loaded_model, data=predict_data, raw_score = True,probability_threshold=0.5)
    #predition
    #取出数据
    prediction_label = predition.iloc[0]['prediction_label']
    # prediction_label
    #显示数据框？

    if prediction_label == 0:
        prediction_score = predition.iloc[0]['prediction_score_0'] * 100
    elif prediction_label == 1:
        prediction_score = predition.iloc[0]['prediction_score_1'] * 100
    
    st.write(f"Predicted Class: {Pathological_diagnosis_dict[prediction_label]}")
    st.write(f"Prediction Probability: {prediction_score}%")

    #生成建议
    if prediction_label == 1:
        advice = (
            f"According to our model, you have a high risk of pathological diagnosis malignant. "
            f"The model predicts that your probability of having heart disease is {prediction_score:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "We recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    elif prediction_label == 0:
        advice = (
            f"According to our model, you have a low risk of pathological diagnosis malignant. "
            f"The model predicts that your probability of not having heart disease is {prediction_score:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)


    #SHAP  force plot

    #计算shape value
    #全样本
    explainer = shap.KernelExplainer(loaded_model[-1].predict, data_shap_input_new) 
    shap_values = explainer.shap_values(data_shap_input_new,n_jobs=-2)
    #输出shap.Explanation对象
    #shap_values2 = explainer(data_shap_input_new) 

    #force_plot
    shap.force_plot(explainer.expected_value, shap_values[-1], data_shap_input_new.iloc[-1,:],matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")





