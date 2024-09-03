import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

df_hitter = pd.read_csv('data\hitter_stats_salary_debut.csv')

def assign_salary_range(row):
  if row < 4500:
    return 0
  elif row < 9000:
    return 1
  elif row < 30000:
    return 2
  else:
    return 3

df_hitter['연봉구간'] = df_hitter['후년연봉'].apply(assign_salary_range)
df_hitter['현재연봉구간'] = df_hitter['연봉(만원)'].apply(assign_salary_range)


# data = df_hitter[['PA','R','H','2B','HR','TB','RBI','BB','HBP','SLG','OBP','WAR','OPS+','연차','현재연봉구간']]
data = df_hitter[['연차','TB', 'H', 'PA', 'RBI','BB', 'R', '2B', 'OBP','OPS+', 'WAR','현재연봉구간']]
# data = df_hitter[['연차','TB', 'H', 'PA', 'RBI','R', 'BB', 'WAR', 'OPS+','현재연봉구간']]

target = df_hitter['연봉구간']

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=42
)

rf_clf = RandomForestClassifier(max_depth = 8,          # 결정 트리 계수
                                min_samples_leaf = 8,   # 각 트리의 최대 깊이
                                min_samples_split = 20, # 리프 노드가 되기 위한 최소 샘플 수
                                n_estimators = 400)     # 내부 노드를 분할하기 위한 최소 샘플 수

rf_clf.fit(X_train,y_train)

# test 데이터 예측
y_pred = rf_clf.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f'Train Accuracy : {rf_clf.score(X_train, y_train)}')
print(f'Test Accuracy  : {rf_clf.score(X_test, y_test)}')
f1 = f1_score(y_test, y_pred, average = 'weighted')
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')

print(f"정확도 : {accuracy}")
print(f"정밀도 : {precision}")
print(f"재현율 : {recall}")

print(f'F1 score : {f1}')