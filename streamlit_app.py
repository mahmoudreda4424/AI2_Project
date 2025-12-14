import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. إعدادات أساسية وثوابت ---
MODEL_PATH = 'ensemble_attrition_model.pkl' # ⚠️ تأكد من أن هذا الملف موجود في نفس مجلد streamlit_app.py
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

# --- 2. تحميل الموديل باستخدام @st.cache_resource ---
# هذا يضمن تحميل الموديل مرة واحدة فقط عند بدء تشغيل التطبيق (مهم جداً للسرعة)

@st.cache_resource
def load_model():
    """تحميل الموديل مرة واحدة."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure '{MODEL_PATH}' is in the correct directory.")
        return None

model = load_model()

# --- 3. دالة معالجة البيانات (تم تكييفها لاستقبال القاموس مباشرةً) ---

def preprocess_input(data_dict):
    """
    يضمن هذا المنطق أن يكون DataFrame النهائي مطابقاً لـ FEATURE_COLS بالضبط.
    المنطق مطابق تماماً لمنطق Flask السابق.
    """
    # 1. إنشاء DataFrame 
    data_df = pd.DataFrame([data_dict])
    
    # 2. إسقاط الأعمدة (Monthly Income)
    # ⚠️ في Streamlit يجب إزالة الـ errors='ignore' أو التأكد من إرسال هذا العمود
    # بما أننا لا نطلب هذا العمود في واجهة المستخدم، سنفترض أنه غير موجود أصلاً في القاموس
    
    # 3. الترميز الثنائي (Gender, Over Time)
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    
    # يجب التعامل مع حالات الأحرف والأسماء المستخدمة في الواجهة
    data_df['Gender'] = data_df['Gender'].map(lambda x: binary_map.get(x))
    data_df['OverTime'] = data_df['OverTime'].map(lambda x: binary_map.get(x))
    
    # 4. الترميز الأحادي الساخن (OHE) - بدون إسقاط Drop First
    OHE_COLS_WITH_SPACES = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    data_df = pd.get_dummies(data_df, columns=OHE_COLS_WITH_SPACES, drop_first=False)
    
    # 5. تنظيف أسماء الأعمدة بعد OHE لمطابقة أسماء الموديل
    data_df.columns = data_df.columns.str.replace(' ', '')
    data_df.columns = data_df.columns.str.replace('-', '_')
    
    # 6. 🛑 النقطة الحاسمة: إعادة الفهرسة لضمان الترتيب الصحيح
    final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return final_df.iloc[0] # نرجع السلسلة (Series) لسهولة العرض في Streamlit

# --- 4. واجهة المستخدم والتفاعل (الـ Frontend) ---

st.set_page_config(
    page_title="Employee Attrition Predictor (ML Model)",
    layout="wide"
)

st.header("👤 نظام التنبؤ بمعدل ترك الموظفين")
st.markdown("يرجى إدخال بيانات الموظف للتنبؤ بمدى احتمالية تركه للعمل، باستخدام النموذج المُدرب على 43 ميزة.")

if model is None:
    st.stop() # إيقاف التطبيق إذا فشل تحميل الموديل

# استخدام st.form لتجميع المدخلات وضمان إرسالها دفعة واحدة
with st.form("attrition_form"):
    
    # تقسيم المدخلات على أعمدة Streamlit لتحسين التصميم
    col1, col2, col3 = st.columns(3)
    
    # --- قسم البيانات الشخصية ---
    with col1:
        st.subheader("معلومات أساسية")
        age = st.slider("العمر (Age)", 18, 60, 30)
        gender = st.selectbox("الجنس (Gender)", ["Male", "Female"])
        marital_status = st.selectbox("الحالة الاجتماعية (Marital Status)", ["Single", "Married", "Divorced"])
        distance_from_home = st.number_input("المسافة من المنزل (بالأميال)", 1, 30, 5)
        
    # --- قسم العمل والراتب ---
    with col2:
        st.subheader("بيانات العمل")
        job_role = st.selectbox("الدور الوظيفي (Job Role)", [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
            'Manufacturing Director', 'Healthcare Representative', 'Manager', 
            'Sales Representative', 'Research Director', 'Human Resources', 
            'Technical Degree', 'Other'
        ], index=0) 
        department = st.selectbox("القسم (Department)", ["Research & Development", "Sales", "Human Resources"])
        job_level = st.selectbox("مستوى الوظيفة (Job Level)", [1, 2, 3, 4, 5])
        job_involvement = st.selectbox("المشاركة الوظيفية (Job Involvement)", [1, 2, 3, 4])
        job_satisfaction = st.selectbox("الرضا الوظيفي (Job Satisfaction)", [1, 2, 3, 4])
        
    # --- قسم الخبرة والأداء ---
    with col3:
        st.subheader("الخبرة والأداء")
        total_working_years = st.number_input("إجمالي سنوات العمل", 0, 40, 5)
        years_at_company = st.number_input("سنوات في الشركة الحالية", 0, 40, 5)
        years_in_current_role = st.number_input("سنوات في الدور الحالي", 0, 18, 2)
        years_with_curr_manager = st.number_input("سنوات مع المدير الحالي", 0, 17, 2)
        over_time = st.selectbox("العمل الإضافي (Over Time)", ["Yes", "No"])
    
    # --- الأزرار وتقديم النموذج ---
    st.markdown("---")
    
    # ⚠️ تم دمج جميع المدخلات هنا في قاموس واحد
    # لاحظ أن أسماء المفاتيح (Keys) هنا يجب أن تتطابق مع الأسماء المستخدمة في دالة preprocess_input
    
    submitted = st.form_submit_button("إجراء التنبؤ")
    
    if submitted:
        
        # 5. تجميع البيانات في قاموس كما كان Flask يستقبله
        input_data = {
            # الأرقام
            'Age': age, 
            'DailyRate': 1000, # قيمة افتراضية أو يمكن طلبها
            'DistanceFromHome': distance_from_home, 
            'Education': 3, # قيمة افتراضية أو يمكن طلبها
            'EnvironmentSatisfaction': 3, # قيمة افتراضية أو يمكن طلبها
            'HourlyRate': 65, # قيمة افتراضية أو يمكن طلبها
            'JobInvolvement': job_involvement, 
            'JobLevel': job_level, 
            'JobSatisfaction': job_satisfaction, 
            'MonthlyRate': 12000, # قيمة افتراضية أو يمكن طلبها
            'NumCompaniesWorked': 1, # قيمة افتراضية أو يمكن طلبها
            'PercentSalaryHike': 12, # قيمة افتراضية أو يمكن طلبها
            'PerformanceRating': 3, # قيمة افتراضية أو يمكن طلبها
            'RelationshipSatisfaction': 3, # قيمة افتراضية أو يمكن طلبها
            'StockOptionLevel': 1, # قيمة افتراضية أو يمكن طلبها
            'TotalWorkingYears': total_working_years, 
            'TrainingTimesLastYear': 2, # قيمة افتراضية أو يمكن طلبها
            'WorkLifeBalance': 3, # قيمة افتراضية أو يمكن طلبها
            'YearsAtCompany': years_at_company, 
            'YearsInCurrentRole': years_in_current_role, 
            'YearsSinceLastPromotion': 1, # قيمة افتراضية أو يمكن طلبها
            'YearsWithCurrManager': years_with_curr_manager, 
            
            # التصنيفات (Categorical)
            'Gender': gender, 
            'OverTime': over_time, 
            'BusinessTravel': 'Travel_Rarely', # قيمة افتراضية أو يمكن طلبها
            'Department': department,
            'EducationField': 'Life Sciences', # قيمة افتراضية أو يمكن طلبها
            'JobRole': job_role, 
            'MaritalStatus': marital_status
        }
        
        # 6. معالجة البيانات
        try:
            processed_series = preprocess_input(input_data)
            processed_df = pd.DataFrame([processed_series])
            
            # 7. التنبؤ
            probability = model.predict_proba(processed_df)[0][1]
            prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
            
            # 8. عرض النتائج
            st.markdown("## 📈 نتيجة التنبؤ")
            
            if prediction == 1:
                st.error(f"**احتمالية ترك العمل (Attrition Probability):** {probability * 100:.2f}%")
                st.warning("😟 بناءً على البيانات المدخلة، الموظف **من المحتمل أن يترك** العمل.")
            else:
                st.success(f"**احتمالية ترك العمل (Attrition Probability):** {probability * 100:.2f}%")
                st.info("😊 بناءً على البيانات المدخلة، الموظف **من المحتمل أن يبقى** في العمل.")
                
            st.markdown(f"> *ملاحظة: تم استخدام عتبة (Threshold) قدرها **{OPTIMAL_THRESHOLD}** لاتخاذ القرار.*")
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة أو التنبؤ: {e}")