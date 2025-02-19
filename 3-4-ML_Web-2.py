# %%
from pycaret.classification import *
import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import random
import pandas as pd

# %% 页面配置
st.set_page_config(page_title="Thyroid Diagnosis", layout="wide")

# %% 标题
st.title("Thyroid Nodule Diagnosis")
st.header("AI-powered Pathological Prediction System")

# %% 固定随机种子
np.random.seed(42)
random.seed(42)

# %% 模型加载
with st.spinner("Loading medical AI model..."):
    loaded_model = load_model(r'C:\Users\32537\Desktop\MLtest\Ada Boost Classifier')
    pipeline = loaded_model[:-1]

# %% 数据准备
data = pd.read_excel(r'3-0-training_data_use.xlsx')
data_shap_input = pipeline.transform(data).drop(columns='Pathological_diagnosis')

# %% 字典配置
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

# %% 输入表单
with st.container(border=True):
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Basic Information")
        Gender = st.selectbox(
            label = 'Gender',
            options = ('Male', 'Female'),
            index = 1,
            help = 'Patient biological gender'
        )
        Age = st.number_input(
            label = 'Age (years)',
            # min_value = 1,
            # max_value = 120,
            value = 45,
            format = '%d',
            help = 'Patient age in years'
        )
        BMI = st.number_input(
            label = 'BMI (Kg/m²)',
            # min_value = 10.0,
            # max_value = 50.0,
            value = 22,
            # step = 0.1,
            # format = '%.1f',
            help = 'Body Mass Index'
        )
        
    with col2:
        st.markdown("### Imaging Features")
        Halo_sign = st.selectbox(
            label = 'Halo Sign',
            options = ('absent', 'Exists'),
            index = 0,
            format_func = str,
            help = 'Perinodular halo appearance'
        )
        Posterior_echo = st.selectbox(
            label = 'Posterior Echo',
            options = ('Absent_of_shadowing', 'Shadowing', 'Posterior_attenuation'),
            index = 0,
            format_func = str,
            help = 'Posterior acoustic features'
        )
        Intra_BFS = st.selectbox(
            label = 'Intra-nodular Blood Flow',
            options = ('Absent', 'Less', 'Rich'),
            index = 0,
            format_func = str,
            help = 'Intranodular vascularity'
        )
        Peri_BFS = st.selectbox(
            label = 'Peri-nodular Blood Flow',
            options = ('Absent', 'Less', 'Rich'),
            index = 0,
            format_func = str,
            help = 'Perinodular vascularity'
        )
                
    with col3:
        st.markdown("### Nodule Characteristics")
        Location = st.selectbox(
            label = 'Location',
            options = ('Right_lobe', 'Left_lobe', 'Isthmus'),
            index = 0,
            format_func = str,
            help = 'Nodule anatomical location'
        )
        Maximum_diameter = st.number_input(
            label = 'Maximum Diameter (mm)',
            # min_value = 0.1,
            # max_value = 100.0,
            value = 0.5,
            step = 0.1,
            format = '%.1f',
            help = 'Largest nodule diameter'
        )
        Composition = st.selectbox(
            label = 'Composition',
            options = ('Others', 'Solid'),
            index = 0,
            format_func = str,
            help = 'Nodule internal composition'
        )
        Shape = st.selectbox(
            label = 'Shape',
            options = ('Microlobulated', 'Others'),
            index = 0,
            format_func = str,
            help = 'Nodule shape characteristics'
        )
        Echogenicity = st.selectbox(  # 添加 Echogenicity 选择框
            label = 'Echogenicity',
            options = ('Others', 'Hypoechogenicity'),
            index = 1,
            format_func = str,
            help = 'Nodule echogenicity characteristics'
        )

        Echogenic_foci = st.selectbox(
            label = 'Echogenic Foci',
            options = ('Others', 'Microcalcification'),
            index = 1,
            format_func = str,
            help = 'Nodule echogenic foci'
        )

        # 添加 Margin 选择框
        Margin = st.selectbox(
            label = 'Margin',
            options = ('Smooth', 'Irregular'),
            index = 0,
            format_func = str,
            help = 'Nodule border characteristics'
        )
        ATR = st.selectbox(
            label = 'Aneurysm Tortuosity Ratio',
            options = ('Wider_than_tall', 'Taller_than_wide'),
            index = 0,
            format_func = str,
            help = 'Aneurysm tortuosity ratio'
        )

# %% 预测执行
if st.button("Start Diagnosis", type="primary"):
    with st.status("Analyzing...", state="running") as status:
        # 数据转换
        st.write("Processing input data...")
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
        
        # 模型预测
        st.write("Running AI diagnosis...")
        predict_data_trans = pipeline.transform(predict_data)
        prediction = predict_model(loaded_model, data=predict_data, 
                                 raw_score=True, probability_threshold=0.5)
        
        # 结果处理
        prediction_label = prediction.iloc[0]['prediction_label']
        if prediction_label == 0:
            prediction_score = prediction.iloc[0]['prediction_score_0'] * 100
        elif prediction_label == 1:
            prediction_score = prediction.iloc[0]['prediction_score_1'] * 100
       
        # 显示结果
        status.update(label="Analysis Complete!", state="complete")
        
    # 结果展示
    with st.container(border=True):
        if prediction_label == 1:
            st.error(f"Malignant Risk: {prediction_score:.1f}%")
            st.markdown("""
            ❗ **Clinical Recommendation**  
            This result suggests significant malignancy risk. Please:
            - Schedule immediate consultation with endocrinologist
            - Consider FNA biopsy
            - Monitor nodule evolution closely
            """)
        elif prediction_label == 0:
            st.success(f"Benign Probability: {prediction_score:.1f}%")
            st.markdown("""
            ✅ **Clinical Recommendation**  
            While low risk, we recommend:
            - Annual ultrasound follow-up
            - Monitor for size/feature changes
            - Report new symptoms promptly
            """)
 
    
    # SHAP 解释
    with st.status("Generating explainability visualization...", state="running") as status:
        # 初始化进度条 自动展示

        progress_bar = st.progress(0)
        status_message = st.empty()

        # 步骤 1: 数据准备
        status_message.write("Preparing data for SHAP analysis...")
        data_shap_input_new = pd.concat([data_shap_input, predict_data_trans], axis=0)
        progress_bar.progress(20)  # 更新进度条

        # 步骤 2: 创建解释器
        status_message.write("Creating SHAP explainer...")
        explainer = shap.KernelExplainer(loaded_model[-1].predict, data_shap_input_new)
        progress_bar.progress(40)  # 更新进度条

        # 步骤 3: 计算 SHAP 值
        status_message.write("Calculating SHAP values (this may take a while)...")
        shap_values = explainer.shap_values(data_shap_input_new, n_jobs=-2)
        progress_bar.progress(80)  # 更新进度条

        # 步骤 4: 生成可视化
        status_message.write("Generating force plot...")
        plt.figure(figsize=(10, 4), dpi=1000)
        shap.force_plot(
            explainer.expected_value,
            shap_values[-1],
            data_shap_input_new.iloc[-1, :],
            matplotlib=True
        )
        plt.title("Feature Impact Visualization", fontsize=14)
        plt.gcf().set_facecolor('#f0f2f6')  # 匹配页面背景色
        progress_bar.progress(100)  # 更新进度条

        # 完成状态
        status.update(label="SHAP analysis complete!", state="complete")

    # 显示可视化结果
    st.pyplot(plt.gcf(), bbox_inches='tight')

    # 添加解释说明
    st.info("ℹ️ The force plot shows how each feature contributes to the prediction. Red indicates increased risk, blue indicates protective factors.")







