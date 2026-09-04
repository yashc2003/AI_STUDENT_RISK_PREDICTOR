import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 2000

student_ids = [f"STU{i:04d}" for i in range(1, n_samples + 1)]

attendance = np.concatenate([
    np.random.normal(85, 8, 700),   # Low risk students
    np.random.normal(65, 12, 800),  # Medium risk students
    np.random.normal(45, 15, 500)   # High risk students
])
attendance = np.clip(attendance, 0, 100)

internal_marks = np.concatenate([
    np.random.normal(78, 8, 700),
    np.random.normal(58, 12, 800),
    np.random.normal(38, 15, 500)
])
internal_marks = np.clip(internal_marks, 0, 100)

prev_cgpa = np.concatenate([
    np.random.normal(8.2, 0.8, 700),
    np.random.normal(6.5, 1.0, 800),
    np.random.normal(4.8, 1.2, 500)
])
prev_cgpa = np.clip(prev_cgpa, 0, 10)

assignment_score = np.concatenate([
    np.random.normal(82, 8, 700),
    np.random.normal(62, 12, 800),
    np.random.normal(40, 15, 500)
])
assignment_score = np.clip(assignment_score, 0, 100)

assignment_submission = np.concatenate([
    np.random.normal(92, 5, 700),
    np.random.normal(72, 15, 800),
    np.random.normal(45, 20, 500)
])
assignment_submission = np.clip(assignment_submission, 0, 100)

study_hours = np.concatenate([
    np.random.normal(4.2, 1.0, 700),
    np.random.normal(2.8, 1.0, 800),
    np.random.normal(1.2, 0.8, 500)
])
study_hours = np.clip(study_hours, 0, 10)

backlogs = np.concatenate([
    np.random.poisson(0.2, 700),
    np.random.poisson(1.0, 800),
    np.random.poisson(2.5, 500)
])
backlogs = np.clip(backlogs, 0, 5)

sleep_hours = np.random.normal(7, 1.5, n_samples)
sleep_hours = np.clip(sleep_hours, 3, 12)

prev_failures = np.concatenate([
    np.random.poisson(0.1, 700),
    np.random.poisson(0.5, 800),
    np.random.poisson(1.8, 500)
])
prev_failures = np.clip(prev_failures, 0, 4)

np.random.shuffle(attendance)
np.random.shuffle(internal_marks)
np.random.shuffle(prev_cgpa)
np.random.shuffle(assignment_score)
np.random.shuffle(assignment_submission)
np.random.shuffle(study_hours)
np.random.shuffle(backlogs)
np.random.shuffle(sleep_hours)
np.random.shuffle(prev_failures)

risk_score = (
    (100 - attendance) * 0.22 +
    (100 - internal_marks) * 0.22 +
    (10 - prev_cgpa) * 10 * 0.18 +
    (100 - assignment_score) * 0.15 +
    (100 - assignment_submission) * 0.10 +
    (10 - study_hours) * 5 * 0.13 +
    backlogs * 12 * 0.12 +
    prev_failures * 12 * 0.12
)

noise = np.random.normal(0, 6, n_samples)
risk_score = risk_score + noise

risk_category = np.zeros(n_samples, dtype=int)
risk_category[risk_score <= 28] = 0  # Low Risk
risk_category[(risk_score > 28) & (risk_score <= 52)] = 1  # Medium Risk
risk_category[risk_score > 52] = 2  # High Risk

df = pd.DataFrame({
    'Student_ID': student_ids,
    'Attendance': attendance.round(1),
    'Internal_Marks': internal_marks.round(1),
    'Previous_CGPA': prev_cgpa.round(2),
    'Assignment_Score': assignment_score.round(1),
    'Assignment_Submission': assignment_submission.round(1),
    'Study_Hours': study_hours.round(1),
    'Backlogs': backlogs,
    'Sleep_Hours': sleep_hours.round(1),
    'Previous_Failures': prev_failures,
    'Risk_Category': risk_category
})

risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
df['Risk_Label'] = df['Risk_Category'].map(risk_labels)

df.to_csv('student_data.csv', index=False)

print(f"Dataset generated with {n_samples} samples")
print("\nRisk Category Distribution:")
print(df['Risk_Label'].value_counts())
print("\nSample data:")
print(df.head())
print("\nFeature statistics:")
print(df.describe())