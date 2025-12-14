import joblib
import pandas as pd
import numpy as np
import streamlit as st

# --- إعدادات أساسية ---
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

# --- تكوين الصفحة ---
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👔",
    layout="wide"
)

# --- تحميل الموديل ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {e}")
        return None

model = load_model()

# --- معالجة البيانات ---
def preprocess_input(data_dict):
    """
    يضمن هذا المنطق أن يكون DataFrame النهائي مطابقاً لـ FEATURE_COLS بالضبط.
    """
    # 1. إنشاء DataFrame مباشرة بأسماء مطابقة للموديل الأصلي
    data_df = pd.DataFrame([{
        'Age': data_dict['Age'],
        'DailyRate': data_dict['Daily Rate'],
        'DistanceFromHome': data_dict['Distance From Home'],
        'Education': data_dict['Education'],
        'EnvironmentSatisfaction': data_dict['Environment Satisfaction'],
        'HourlyRate': data_dict['Hourly Rate'],
        'JobInvolvement': data_dict['Job Involvement'],
        'JobLevel': data_dict['Job Level'],
        'JobSatisfaction': data_dict['Job Satisfaction'],
        'MonthlyRate': data_dict['Monthly Rate'],
        'NumCompaniesWorked': data_dict['Num Companies Worked'],
        'PercentSalaryHike': data_dict['Percent Salary Hike'],
        'PerformanceRating': data_dict['Performance Rating'],
        'RelationshipSatisfaction': data_dict['Relationship Satisfaction'],
        'StockOptionLevel': data_dict['Stock Option Level'],
        'TotalWorkingYears': data_dict['Total Working Years'],
        'TrainingTimesLastYear': data_dict['Training Times Last Year'],
        'WorkLifeBalance': data_dict['Work Life Balance'],
        'YearsAtCompany': data_dict['Years At Company'],
        'YearsInCurrentRole': data_dict['Years In Current Role'],
        'YearsSinceLastPromotion': data_dict['Years Since Last Promotion'],
        'YearsWithCurrManager': data_dict['Years With Curr Manager'],
        'Gender': data_dict['Gender'],
        'Over Time': data_dict['Over Time'],
        'Business Travel': data_dict['Business Travel'],
        'Department': data_dict['Department'],
        'Education Field': data_dict['Education Field'],
        'Job Role': data_dict['Job Role'],
        'Marital Status': data_dict['Marital Status']
    }])
    
    # 2. الترميز الثنائي (Gender, Over Time)
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    data_df['Gender'] = data_df['Gender'].map(lambda x: binary_map.get(x, 0))
    data_df['Over Time'] = data_df['Over Time'].map(lambda x: binary_map.get(x, 0))
    
    # إعادة تسمية Over Time
    data_df = data_df.rename(columns={'Over Time': 'OverTime'})

    # 3. الترميز الأحادي الساخن (OHE) مع المسافات كما في الكود الأصلي
    OHE_COLS_WITH_SPACES = ['Business Travel', 'Department', 'Education Field', 'Job Role', 'Marital Status']
    data_df = pd.get_dummies(data_df, columns=OHE_COLS_WITH_SPACES, drop_first=False)
    
    # 4. تنظيف أسماء الأعمدة بعد OHE لمطابقة أسماء الموديل (بنفس الطريقة الأصلية)
    data_df.columns = data_df.columns.str.replace(' ', '')
    data_df.columns = data_df.columns.str.replace('-', '_')
    
    # 5. 🛑 النقطة الحاسمة: إعادة الفهرسة لضمان الترتيب الصحيح
    final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return final_df

# --- واجهة المستخدم ---
st.title("👔 نظام التنبؤ بترك الموظفين للعمل")
st.markdown("---")

if model is None:
    st.error("⚠️ لم يتم تحميل النموذج بشكل صحيح. يرجى التحقق من وجود الملف.")
    st.stop()

# إنشاء أعمدة للتنسيق
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 البيانات الديموغرافية")
    age = st.number_input("العمر (Age)", min_value=18, max_value=65, value=30)
    gender = st.selectbox("الجنس (Gender)", ["Male", "Female"])
    marital_status = st.selectbox("الحالة الاجتماعية (Marital Status)", 
                                  ["Single", "Married", "Divorced"])
    distance = st.number_input("المسافة من المنزل (Distance From Home)", 
                              min_value=1, max_value=50, value=10)

with col2:
    st.subheader("💼 معلومات الوظيفة")
    department = st.selectbox("القسم (Department)", 
                             ["Research & Development", "Sales", "Human Resources"])
    job_role = st.selectbox("الدور الوظيفي (Job Role)", 
                           ["Sales Executive", "Research Scientist", "Laboratory Technician",
                            "Manufacturing Director", "Healthcare Representative", "Manager",
                            "Sales Representative", "Research Director", "Human Resources"])
    job_level = st.slider("المستوى الوظيفي (Job Level)", 1, 5, 2)
    job_involvement = st.slider("المشاركة الوظيفية (Job Involvement)", 1, 4, 3)
    job_satisfaction = st.slider("الرضا الوظيفي (Job Satisfaction)", 1, 4, 3)

with col3:
    st.subheader("💰 المعلومات المالية")
    monthly_income = st.number_input("الدخل الشهري (Monthly Income)", 
                                    min_value=1000, max_value=20000, value=5000)
    hourly_rate = st.number_input("الأجر بالساعة (Hourly Rate)", 
                                 min_value=30, max_value=100, value=65)
    daily_rate = st.number_input("الأجر اليومي (Daily Rate)", 
                                min_value=100, max_value=1500, value=800)
    monthly_rate = st.number_input("المعدل الشهري (Monthly Rate)", 
                                  min_value=2000, max_value=27000, value=14000)
    percent_salary_hike = st.number_input("نسبة زيادة الراتب (Percent Salary Hike)", 
                                         min_value=11, max_value=25, value=15)

# أعمدة إضافية
col4, col5 = st.columns(2)

with col4:
    st.subheader("🎓 التعليم والخبرة")
    education = st.slider("مستوى التعليم (Education)", 1, 5, 3,
                         help="1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor")
    education_field = st.selectbox("مجال التعليم (Education Field)",
                                  ["Life Sciences", "Medical", "Marketing", 
                                   "Technical Degree", "Other", "Human Resources"])
    total_working_years = st.number_input("إجمالي سنوات العمل (Total Working Years)", 
                                         min_value=0, max_value=40, value=10)
    num_companies_worked = st.number_input("عدد الشركات السابقة (Num Companies Worked)", 
                                          min_value=0, max_value=9, value=2)
    training_times = st.number_input("مرات التدريب السنة الماضية (Training Times Last Year)", 
                                    min_value=0, max_value=6, value=2)

with col5:
    st.subheader("⏰ تفاصيل العمل")
    years_at_company = st.number_input("سنوات في الشركة (Years At Company)", 
                                      min_value=0, max_value=40, value=5)
    years_in_role = st.number_input("سنوات في الدور الحالي (Years In Current Role)", 
                                   min_value=0, max_value=18, value=3)
    years_since_promotion = st.number_input("سنوات منذ آخر ترقية (Years Since Last Promotion)", 
                                           min_value=0, max_value=15, value=1)
    years_with_manager = st.number_input("سنوات مع المدير الحالي (Years With Curr Manager)", 
                                        min_value=0, max_value=17, value=3)
    overtime = st.selectbox("العمل الإضافي (Over Time)", ["No", "Yes"])
    business_travel = st.selectbox("السفر للعمل (Business Travel)", 
                                  ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# المزيد من الحقول
col6, col7 = st.columns(2)

with col6:
    st.subheader("😊 مستويات الرضا")
    environment_satisfaction = st.slider("الرضا عن البيئة (Environment Satisfaction)", 1, 4, 3)
    relationship_satisfaction = st.slider("الرضا عن العلاقات (Relationship Satisfaction)", 1, 4, 3)
    work_life_balance = st.slider("التوازن بين العمل والحياة (Work Life Balance)", 1, 4, 3)

with col7:
    st.subheader("📈 الأداء والمكافآت")
    performance_rating = st.slider("تقييم الأداء (Performance Rating)", 3, 4, 3)
    stock_option_level = st.slider("مستوى خيارات الأسهم (Stock Option Level)", 0, 3, 1)

st.markdown("---")

# زر التنبؤ
if st.button("🔮 التنبؤ بقرار الموظف", type="primary", use_container_width=True):
    # جمع البيانات
    data_dict = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Distance From Home': distance,
        'Department': department,
        'Job Role': job_role,
        'Job Level': job_level,
        'Job Involvement': job_involvement,
        'Job Satisfaction': job_satisfaction,
        'Monthly Income': monthly_income,
        'Hourly Rate': hourly_rate,
        'Daily Rate': daily_rate,
        'Monthly Rate': monthly_rate,
        'Percent Salary Hike': percent_salary_hike,
        'Education': education,
        'Education Field': education_field,
        'Total Working Years': total_working_years,
        'Num Companies Worked': num_companies_worked,
        'Training Times Last Year': training_times,
        'Years At Company': years_at_company,
        'Years In Current Role': years_in_role,
        'Years Since Last Promotion': years_since_promotion,
        'Years With Curr Manager': years_with_manager,
        'Over Time': overtime,
        'Business Travel': business_travel,
        'Environment Satisfaction': environment_satisfaction,
        'Relationship Satisfaction': relationship_satisfaction,
        'Work Life Balance': work_life_balance,
        'Performance Rating': performance_rating,
        'Stock Option Level': stock_option_level
    }
    
    try:
        # معالجة البيانات
        processed_data = preprocess_input(data_dict)
        
        # التنبؤ باستخدام numpy array بدلاً من DataFrame لتجنب مشكلة أسماء الأعمدة
        probability = model.predict_proba(processed_data.values)[0][1]
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        
        # عرض النتائج
        st.markdown("---")
        st.subheader("📊 نتيجة التنبؤ")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 1:
                st.error("### 😟 من المحتمل أن يترك العمل")
            else:
                st.success("### 😊 من المحتمل أن يبقى")
        
        with result_col2:
            st.metric("احتمالية الترك", f"{probability:.2%}")
        
        with result_col3:
            st.info(f"العتبة المستخدمة: {OPTIMAL_THRESHOLD}")
        
        # شريط التقدم
        st.progress(probability)
        
        # توصيات
        if prediction == 1:
            st.warning("⚠️ **توصيات للحد من ترك الموظف:**")
            recommendations = []
            if job_satisfaction < 3:
                recommendations.append("- تحسين الرضا الوظيفي")
            if work_life_balance < 3:
                recommendations.append("- تحسين التوازن بين العمل والحياة")
            if years_since_promotion > 3:
                recommendations.append("- النظر في فرص الترقية")
            if overtime == "Yes":
                recommendations.append("- تقليل ساعات العمل الإضافية")
            if environment_satisfaction < 3:
                recommendations.append("- تحسين بيئة العمل")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.write("- مراجعة شاملة لظروف العمل والتواصل مع الموظف")
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء التنبؤ: {str(e)}")

# معلومات إضافية في الشريط الجانبي
with st.sidebar:
    st.header("ℹ️ معلومات النظام")
    st.write("هذا النظام يستخدم نموذج Machine Learning للتنبؤ بما إذا كان الموظف سيترك العمل أم لا.")
    st.write(f"**العتبة المثلى:** {OPTIMAL_THRESHOLD}")
    st.write(f"**عدد الميزات:** {len(FEATURE_COLS)}")
    
    st.markdown("---")
    st.subheader("📝 ملاحظات")
    st.write("- املأ جميع الحقول بدقة")
    st.write("- النتيجة تعتمد على البيانات المدخلة")
    st.write("- استخدم التوصيات لتحسين الاحتفاظ بالموظفين")