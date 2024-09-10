# Bibliotecas Gerais
import pandas as pd
import numpy as np

# Bibliotecas para Plotar Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas para Preparação dos Dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Biblioteca para Classificação
from xgboost import XGBClassifier

# Bibliotecas para Pós Processamento das Classificações
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, precision_recall_curve


# Carregando os dados
train_data_url = 'https://raw.githubusercontent.com/Spogis/SimpleClassification/master/Datasets/train_data.csv'
validation_data_url = 'https://raw.githubusercontent.com/Spogis/SimpleClassification/master/Datasets/test_data.csv'

train_data = pd.read_csv(train_data_url)
validation_data = pd.read_csv(validation_data_url)

# Especificando as colunas categóricas e numéricas
categorical_features = ['Title', 'Sex', 'TicketAppearances', 'CabinPrefix', 'IsAlone', 'Embarked']
numerical_features = ['Pclass', 'Fare', 'FamilySize', 'SibSp', 'Parch']

# Codificação das variáveis categóricas
label_encoder = LabelEncoder()
for feature in categorical_features:
    combined_data = pd.concat([train_data[feature], validation_data[feature]], axis=0)
    combined_data_encoded = label_encoder.fit_transform(combined_data)
    train_data[feature] = combined_data_encoded[:len(train_data)]
    validation_data[feature] = combined_data_encoded[len(train_data):]

# Obter a ordem das colunas para entrada de dados (categóricas e numéricas)
feature_order = categorical_features + numerical_features

# Exibindo a ordem das features para a entrada de dados
print("Ordem de entrada dos dados no modelo:")
print(feature_order)

# Padronização das variáveis numéricas
scaler = StandardScaler()
scaler.fit(train_data[numerical_features])
train_data[numerical_features] = scaler.transform(train_data[numerical_features])
validation_data[numerical_features] = scaler.transform(validation_data[numerical_features])

# Separando as variáveis independentes da variável alvo (Classificação)
y = train_data['Survived']
X = train_data[categorical_features + numerical_features]

# Dividindo os dados em conjuntos de treinamento(80%) e teste(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o modelo XGBClassifier com parâmetros fixos
xgb = XGBClassifier(
    n_estimators=200,      # Número de árvores
    max_depth=5,           # Profundidade máxima da árvore
    learning_rate=0.05,    # Taxa de aprendizado
    subsample=0.8,         # Fração de amostras usadas para cada árvore
    colsample_bytree=0.8,  # Fração de colunas usadas para cada árvore
    min_child_weight=2,    # Peso mínimo das instâncias em cada folha
    use_label_encoder=False,
    eval_metric='logloss'
)

# Treinando o modelo
xgb.fit(X_train, y_train)


########################################################################################################################
# A partir daqui fazemos o pós processamento das análises
########################################################################################################################

# Avaliando o modelo no conjunto de teste
test_accuracy = xgb.score(X_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy*100:.2f}%")

y_pred = xgb.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.2f}")

# Fazendo previsões com o modelo treinado
predictions = xgb.predict(X)

# Calculando e plotando a matriz de confusão
conf_matrix = confusion_matrix(y, predictions)
conf_matrix_percentage_per_class = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True) * 100
annot = np.array([["{:.1f}%".format(val) for val in row] for row in conf_matrix_percentage_per_class])

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percentage_per_class, annot=annot, fmt="", cmap="Blues",
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            square=True,
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 14}, linewidth=.5)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.tight_layout()
# Salvar a figura da matriz de confusão
plt.savefig('confusion_matrix.png', dpi=600)
#plt.show()

# Importância das características
feature_importances = xgb.feature_importances_
features = categorical_features + numerical_features
importances_df = pd.DataFrame({'Features': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=importances_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
# Salvar a figura da importância das características
plt.savefig('feature_importance.png', dpi=600)
#plt.show()

# Calculando as probabilidades para a classe positiva
y_prob = xgb.predict_proba(X_test)[:, 1]

# Calculando a AUC
auc_score = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc_score:.2f}")

# Plotando a curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal de sorte
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('ROC_curve.png', dpi=600)
#plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('Precision_Recall_curve.png', dpi=600)
#plt.show()

########################################################################################################################
# Teste de previsão com 5 linhas aleatórias do dataset original
########################################################################################################################

# Selecionando 5 linhas aleatórias do conjunto de validação
random_rows = validation_data.sample(n=5, random_state=42)

# Obtendo os PassengerId e os nomes das linhas aleatórias
passenger_ids = random_rows['PassengerId'].values
names = random_rows['Name'].values

# Obtendo os valores verdadeiros para comparação
true_values = random_rows['Survived'].values

# Aplicando as mesmas transformações de codificação (LabelEncoder) e padronização (StandardScaler)
# Codificando variáveis categóricas
for feature in categorical_features:
    combined_data = pd.concat([train_data[feature], validation_data[feature]], axis=0)
    combined_data_encoded = label_encoder.fit_transform(combined_data)
    random_rows[feature] = label_encoder.transform(random_rows[feature])

# Padronizando variáveis numéricas
random_rows[numerical_features] = scaler.transform(random_rows[numerical_features])

# Selecionando as colunas de interesse (categóricas e numéricas)
random_rows_X = random_rows[categorical_features + numerical_features]

# Fazendo as previsões com as linhas aleatórias
random_predictions = xgb.predict(random_rows_X)

print()
# Exibindo os resultados
for passenger_id, name, true_value, prediction in zip(passenger_ids, names, true_values, random_predictions):
    print(f"Name: {name}")
    print(f"Valor verdadeiro: {'Sobreviveu' if true_value == 1 else 'Não sobreviveu'}")
    print(f"Predição: {'Sobreviveu' if prediction == 1 else 'Não sobreviveu'}")
    print()


