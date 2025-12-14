import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. إعدادات أساسية وثوابت ---
MODEL_PATH = 'ensemble_attrition_model.pkl' 
OPTIMAL_THRESHOLD = 0.43 

# 🛑 القائمة النهائية والوحيدة الصحيحة للأعمدة الـ 43 بالترتيب الدقيق المطلوب
FEATURE_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 
    'Gender', 'OverTime', 
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
    'EducationField_Other', 'EducationField_Technical Degree', 
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager', 
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 
    'MaritalStatus_Married', 'MaritalStatus_Single'
] 

# --- 2. تحميل الموديل (معالج بنفس الطريقة) ---

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure '{MODEL_PATH}' and all required libraries (e.g., xgboost) are available.")
        return None

model = load_model()

# --- 3. دالة معالجة البيانات (تم تصحيحها) ---

def preprocess_input(data_dict):
    data_df = pd.DataFrame([data_dict])
    
    # 2. (Monthly Income) - تم تركه لأنه غير مطلوب في المدخلات هنا
    
    # 3. الترميز الثنائي
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    data_df['Gender'] = data_df['Gender'].map(lambda x: binary_map.get(x))
    # يجب أن يكون اسم العمود هنا 'OverTime'
    data_df['OverTime'] = data_df['OverTime'].map(lambda x: binary_map.get(x)) 

    # 4. الترميز الأحادي الساخن (OHE)
    OHE_COLS_WITH_SPACES = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    data_df = pd.get_dummies(data_df, columns=OHE_COLS_WITH_SPACES, drop_first=False)
    
    # 5. 🛑 إزالة تنظيف الأعمدة لتجنب تغيير أسماء الأقسام ذات المسافات
    # data_df.columns = data_df.columns.str.replace(' ', '')
    # data_df.columns = data_df.columns.str.replace('-', '_')
    
    # 6. إعادة الفهرسة لضمان الترتيب الصحيح
    final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return final_df.iloc[0], final_df # نرجع السلسلة و الـ DF الكامل

# --- 4. واجهة المستخدم والتفاعل (المحسّنة) ---

st.set_page_config(
    page_title="Employee Attrition Predictor (ML Model)",
    layout="wide",
    initial_sidebar_state="expanded" # فتح الشريط الجانبي تلقائيًا
)

st.title("📊 نظام التنبؤ بترك الموظفين")

if model is None:
    st.error("⚠️ النموذج غير محمل. يرجى مراجعة ملفات النموذج و requirements.txt.")
    st.stop()

# --- الشريط الجانبي (للمدخلات الأقل أهمية أو القيم الافتراضية) ---
with st.sidebar:
    st.header("⚙️ إعدادات الموظف و الرضا")
    
    # المدخلات التي كانت في الأعمدة وتم نقلها
    age = st.slider("العمر (Age)", 18, 60, 30)
    gender = st.selectbox("الجنس (Gender)", ["Male", "Female"])
    
    # قيم افتراضية تم طلبها
    st.markdown("---")
    st.markdown("**مستويات الرضا والتقييم (1=منخفض, 4=مرتفع)**")
    environment_satisfaction = st.selectbox("الرضا عن البيئة (Environment Satisfaction)", [1, 2, 3, 4], index=2) # القيمة الافتراضية 3
    job_satisfaction = st.selectbox("الرضا الوظيفي (Job Satisfaction)", [1, 2, 3, 4], index=2) # القيمة الافتراضية 3
    performance_rating = st.selectbox("تقييم الأداء (Performance Rating)", [1, 2, 3, 4], index=2) # القيمة الافتراضية 3
    
    # القيم الثابتة في الكود (يمكن جعلها مدخلات متقدمة)
    daily_rate = 1000
    hourly_rate = 65
    monthly_rate = 12000
    education = 3
    num_companies_worked = 1
    percent_salary_hike = 12
    relationship_satisfaction = 3
    stock_option_level = 1
    training_times_last_year = 2
    work_life_balance = 3
    years_since_last_promotion = 1


# --- الواجهة الرئيسية (للمدخلات الأكثر أهمية) ---
with st.form("attrition_form"):
    
    st.subheader("معلومات العمل الأساسية والخبرة")
    
    col1, col2, col3 = st.columns(3)
    
    # --- قسم البيانات الوظيفية ---
    with col1:
        department = st.selectbox("القسم (Department)", ["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox("الدور الوظيفي (Job Role)", [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
            'Manufacturing Director', 'Healthcare Representative', 'Manager', 
            'Sales Representative', 'Research Director', 'Human Resources', 
            'Technical Degree', 'Other'
        ]) 
        job_level = st.selectbox("مستوى الوظيفة (Job Level)", [1, 2, 3, 4, 5])
        job_involvement = st.selectbox("المشاركة الوظيفية (Job Involvement)", [1, 2, 3, 4])
        
    # --- قسم الخبرة والمدة ---
    with col2:
        total_working_years = st.number_input("إجمالي سنوات العمل", 0, 40, 5)
        years_at_company = st.number_input("سنوات في الشركة الحالية", 0, 40, 5)
        years_in_current_role = st.number_input("سنوات في الدور الحالي", 0, 18, 2)
        years_with_curr_manager = st.number_input("سنوات مع المدير الحالي", 0, 17, 2)
        
    # --- قسم عوامل أخرى ---
    with col3:
        marital_status = st.selectbox("الحالة الاجتماعية (Marital Status)", ["Single", "Married", "Divorced"])
        distance_from_home = st.number_input("المسافة من المنزل (بالأميال)", 1, 30, 5)
        over_time = st.selectbox("العمل الإضافي (Over Time)", ["Yes", "No"])
        
        # اختيار السفر
        business_travel = st.selectbox("سفر العمل (Business Travel)", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

        # التعليم
        education_field = st.selectbox("مجال التعليم (Education Field)", ['Life Sciences', 'Marketing', 'Medical', 'Technical Degree', 'Human Resources', 'Other'])
    
    st.markdown("---")
    submitted = st.form_submit_button("🚀 إجراء التنبؤ", type="primary") # زر بارز
    
    if submitted:
        
        input_data = {
            'Age': age, 'DailyRate': daily_rate, 'DistanceFromHome': distance_from_home, 
            'Education': education, 'EnvironmentSatisfaction': environment_satisfaction, 
            'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement, 
            'JobLevel': job_level, 'JobSatisfaction': job_satisfaction, 
            'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_companies_worked, 
            'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating, 
            'RelationshipSatisfaction': relationship_satisfaction, 'StockOptionLevel': stock_option_level, 
            'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times_last_year, 
            'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company, 
            'YearsInCurrentRole': years_in_current_role, 'YearsSinceLastPromotion': years_since_last_promotion, 
            'YearsWithCurrManager': years_with_curr_manager, 
            
            'Gender': gender, 'OverTime': over_time, 
            'BusinessTravel': business_travel, 
            'Department': department,
            'EducationField': education_field, 
            'JobRole': job_role, 
            'MaritalStatus': marital_status
        }
        
        try:
            processed_series, processed_df = preprocess_input(input_data)
            
            probability = model.predict_proba(processed_df)[0][1]
            prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
            
            st.markdown("### 🎯 النتيجة والتحليل")
            
            # --- عرض النتائج بشكل Metrics ---
            col_res_1, col_res_2 = st.columns([1, 2])
            
            with col_res_1:
                # عرض الاحتمالية بشكل Metric
                st.metric(label="احتمالية المغادرة (Attrition Probability)", value=f"{probability * 100:.2f}%")

            with col_res_2:
                # عرض القرار النهائي
                if prediction == 1:
                    st.error("❌ **القرار:** الموظف **مُعرَّض لخطر الترك** (Risk of Attrition)")
                else:
                    st.success("✅ **القرار:** الموظف **من المحتمل أن يبقى** (Likely to Stay)")

            # --- عرض التفاصيل ---
            st.markdown("---")
            with st.expander("📊 تفاصيل الإدخال والنموذج"):
                st.markdown(f"**عتبة القرار المستخدمة:** {OPTIMAL_THRESHOLD}")
                st.dataframe(processed_df.T, use_container_width=True) # عرض البيانات المعالجة
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة أو التنبؤ: {e}")