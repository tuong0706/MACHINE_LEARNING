import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pprint import pprint

dataFrame = pd.read_csv(os.path.join(os.path.dirname(__file__), "winequality-red.csv"))

#~ In ra 5 hàng đầu tiên của DataFrame bằng dataFrame.head().
print("     "*14+"----- In ra 5 hàng đầu tiên -----")
print(dataFrame.head())
print("")

#~ Tính các thống kê mô tả cơ bản cho từng cột trong dữ liệu
print("     "*8+"----- Thống kê mô tả cơ bản -----")
print(dataFrame.describe().T)
print("")

#~ Kiểm tra và hiển thị số lượng giá trị thiếu (missing values) trong DataFrame `dataFrame`.
print("     "*9+"----- Kiểm tra và hiển thị số lượng giá trị thiếu -----")
print(dataFrame.isnull().sum())
print("")

from scipy import stats
#~ Tính z-score cho mỗi giá trị trong DataFrame `dataFrame`
print("     "*11+"----- Tính z-score -----")
z = stats.zscore(dataFrame)
z.to_csv('Machine_Learning_Project/Z-score.csv')
print(z)
print("")

#~ Loại bỏ các ngoại lai
print("     "*11+"----- Loại bỏ các ngoại lai dựa trên z-score -----")
threshold = 3
print(np.where(z > 3))

dataFrame_o = dataFrame[(z < 3).all(axis=1)] 
print(dataFrame.shape) 

dataFrame_o.to_csv('Machine_Learning_Project/new_dataFrame.csv')
print(dataFrame_o.shape) 
print("")

from sklearn.model_selection import train_test_split
#~ Chuẩn bị dữ liệu để huấn luyện
print("     "*11+"----- Chuẩn bị dữ liệu -----")

X = dataFrame_o.drop(columns = 'chất lượng')
y = dataFrame_o['chất lượng']

print(X.head())
print("")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

print('Accuracy: ', metrics.accuracy_score(y_test,y_pred))

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_, index=dataFrame_o.columns[:11]).sort_values(ascending=False)
print(feature_imp)

#~ Trực quan hóa đặc trưng quan trọng
import seaborn as sns

sns.barplot(x=feature_imp, y=feature_imp.index, palette="hls")

plt.xlabel('Điểm số đặc trưng quan trọng')
plt.ylabel('Đặc trưng')
plt.title("Trực quan hóa đặc trưng quan trọng")
#// plt.legend()
plt.show()

#~ Tinh chỉnh bằng RandomSearchCrossValidation
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 6)
# Thông số được sử dụng
print("     "*11+'----- Các thông số đang được sử dụng -----')
pprint(rf.get_params())
print("")

#~ Thiết lập tham số
from sklearn.model_selection import RandomizedSearchCV
# Số lượng cây trong rừng
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Số lượng đặc trưng mà mô hình sẽ cân nhắc khi xây dựng cây
max_features = ['auto', 'sqrt']

# Độ sâu tối đa của cây quyết định
max_depth = [int(x) for x in np.linspace(2, 14, num = 7)]
max_depth.append(None)

# Số mẫu tối thiểu cần thiết để chia một nút thành hai nút con
min_samples_split = [2, 5, 10]

# Số mẫu tối thiểu cần thiết để tạo một lá
min_samples_leaf = [1, 2, 4]

# Xác định mô hình có sử dụng Bootstrap Sampling hay không
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("     "*11+'----- Thiết lập thông số -----')
pprint(random_grid)


#~ Chọn ra các tỗ hợp tham số tốt nhất
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=6, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)

print("     "*11+'----- Tổ hợp tham số tốt nhất -----')
pprint(rf_random.best_params_)

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Hiệu suất của model')
    print('Giá trị trung bình của sai số: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Độ chính xác = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 6)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test,y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Mức độ cải thiện {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

#~ Thực hiện tìm kiếm siêu tham số 
#Grid Search with Cross Validation

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 10, 12, 14],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_

best_grid = grid_search.best_estimator_
pprint(best_grid)
grid_accuracy = evaluate(best_grid, X_test, y_test)



# Đọc dữ liệu bên ngoài
external_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "external_wine.csv"))  # Thay đổi đường dẫn tới file CSV của dữ liệu bên ngoài

# Hiển thị 5 hàng đầu tiên của dữ liệu bên ngoài
print("     "*14+"----- Dữ liệu bên ngoài -----")
print(external_data.head())
print("")

# Loại bỏ các cột không cần thiết (nếu có)
# Chắc chắn rằng cấu trúc của dữ liệu bên ngoài giống với dữ liệu huấn luyện
external_data = external_data.drop(columns='column_to_drop', errors='ignore')

# Chuẩn bị dữ liệu để dự đoán
X_external = external_data.drop(columns='chất lượng', errors='ignore')

# Dự đoán chất lượng bằng mô hình đã huấn luyện
external_predictions = best_grid.predict(X_external)

# Hiển thị kết quả dự đoán cho dữ liệu bên ngoài
print("     "*14+"----- Kết quả dự đoán cho dữ liệu bên ngoài -----")
print(external_predictions)

# Thêm cột dự đoán vào DataFrame của dữ liệu bên ngoài
external_data['Dự đoán chất lượng'] = external_predictions

# Hiển thị dữ liệu bên ngoài với cột dự đoán
print("     "*14+"----- Dữ liệu bên ngoài với dự đoán -----")
print(external_data)