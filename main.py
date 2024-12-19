import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt




# 1. 数据加载
train_data = pd.read_csv('C://Users//Administrator//Desktop//train.csv')  # 读取训练集
test_A_data = pd.read_csv('C://Users//Administrator//Desktop//testA.csv')  # 读取测试集A

# 2. 数据预处理
# 检查数据类型和缺失值

print(train_data.info())
print(train_data.isnull().sum())

# 填充缺失值
# 对数值型缺失值填充均值
numeric_columns = train_data.select_dtypes(include=[np.number]).columns

# 确保 test_A_data 不包含 'isDefault' 列
numeric_columns_test_A = [col for col in numeric_columns if col in test_A_data.columns]

train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].mean())
test_A_data[numeric_columns_test_A] = test_A_data[numeric_columns_test_A].fillna(test_A_data[numeric_columns_test_A].mean())

# 对类别变量进行编码（假设有一些类别变量，如'category_column'）
label_columns = ['employmentTitle', 'purpose', 'postCode', 'title']  # 替换为实际类别列
# 将类别列进行LabelEncoder编码
for col in label_columns:
    all_data = pd.concat([train_data[col], test_A_data[col]], axis=0)

    encoder = LabelEncoder()
    encoder.fit(all_data.astype(str))  # 用合并后的数据拟合编码器

    train_data[col] = encoder.transform(train_data[col].astype(str))
    test_A_data[col] = encoder.transform(test_A_data[col].astype(str))

# 特征和标签划分
X = train_data.drop(columns=['id', 'isDefault'])  # 'id'为唯一标识符，'isDefault'为标签
y = train_data['isDefault']

# 数据标准化
X = X.select_dtypes(include=['number'])

# 对每个类别列进行LabelEncoder转换
for col in label_columns:
    if col in X.columns:  # 确保该列在 X 中
        X[col] = encoder.fit_transform(X[col].astype(str))

# 填补缺失值（如果有的话）
X = X.fillna(X.mean())

# 使用 StandardScaler 缩放数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 打印结果
print(X_scaled)
# 4. 数据集划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 构建深度神经网络模型
model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # 由于是二分类问题，输出层使用sigmoid激活函数
])

# 6. 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val), verbose=2)

# 8. 模型评估
y_val_pred = model.predict(X_val)
y_val_pred = (y_val_pred > 0.5).astype(int)  # 将概率值转换为0或1
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'ROC AUC on validation set: {roc_auc}')


# 9. 在测试集A上进行预测
X_test_A = test_A_data.drop(columns=['id'])

# 确保 X_test_A 和 X_train 的列名一致
# 使用 train_data 中的列名来确保一致性
X_test_A = X_test_A[X.columns]

# 对测试集进行标准化
X_test_A_scaled = scaler.transform(X_test_A)

# 使用训练好的模型进行预测
y_test_A_pred = model.predict(X_test_A_scaled)

# 将预测结果保存为csv
result = pd.DataFrame({'id': test_A_data['id'], 'isDefault': y_test_A_pred.flatten()})
result.to_csv('submission.csv', index=False)


print("提交文件已经生成：submission.csv")
# 从 history 对象中提取数据
history_dict = history.history
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(train_accuracy) + 1)

# 绘制准确率曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
