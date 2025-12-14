import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

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

# --- تهيئة تطبيق Flask وتحميل الموديل ---
app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

# --- مسار معالجة البيانات ---

def preprocess_input(data_json):
    """
    يضمن هذا المنطق أن يكون DataFrame النهائي مطابقاً لـ FEATURE_COLS بالضبط.
    """
    # 1. إنشاء DataFrame 
    data_df = pd.DataFrame([data_json])
    
    # 2. إسقاط الأعمدة (Monthly Income)
    data_df = data_df.drop('Monthly Income', axis=1, errors='ignore')
    
    # 3. الترميز الثنائي (Gender, Over Time)
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    # يجب أن تتطابق أسماء الأعمدة هنا مع الـ keys في JSON (من الـ form)
    data_df['Gender'] = data_df['Gender'].map(lambda x: binary_map.get(x, 0))
    data_df['Over Time'] = data_df['Over Time'].map(lambda x: binary_map.get(x, 0))

    # 4. الترميز الأحادي الساخن (OHE) - بدون إسقاط Drop First
    OHE_COLS_WITH_SPACES = ['Business Travel', 'Department', 'Education Field', 'Job Role', 'Marital Status']
    data_df = pd.get_dummies(data_df, columns=OHE_COLS_WITH_SPACES, drop_first=False)
    
    # 5. تنظيف أسماء الأعمدة بعد OHE لمطابقة أسماء الموديل
    data_df.columns = data_df.columns.str.replace(' ', '')
    data_df.columns = data_df.columns.str.replace('-', '_')
    
    # 6. 🛑 النقطة الحاسمة: إعادة الفهرسة لضمان الترتيب الصحيح
    final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return final_df

# --- مسار الصفحة الرئيسية (GET) ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- نقطة نهاية التنبؤ (API Endpoint - POST) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    try:
        data = request.get_json(force=True) 
        processed_data = preprocess_input(data)
        
        if processed_data.shape[1] != 43:
             # هذا خطأ لن يظهر طالما FEATURE_COLS صحيحة
             return jsonify({"error": f"Feature count mismatch after processing. Expected 43, got {processed_data.shape[1]}. Please ensure all 28 fields are submitted."}), 400

        # التنبؤ
        probability = model.predict_proba(processed_data)[0][1]
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        result_label = "Likely to leave (Yes) 😟" if prediction == 1 else "Likely to stay (No) 😊"
        
        return jsonify({
            'attrition_prediction': result_label,
            'probability_of_attrition': f"{probability:.4f}",
            'threshold_used': OPTIMAL_THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}. Check that you sent all 28 fields in the correct format (JSON)."}), 400

# --- تشغيل التطبيق ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)