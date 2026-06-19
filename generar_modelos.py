"""
NutriML-Pareja — Regenerar modelos.pkl en tu PC local
Ejecutar UNA SOLA VEZ antes de iniciar la app
"""
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("=" * 50)
print("NutriML-Pareja — Generando modelos locales")
print("=" * 50)

SEED = 42
N    = 424
np.random.seed(SEED)

# ── Dataset calibrado con variables de la tesis ────────────────────────────
X = np.column_stack([
    np.clip(np.random.normal(26.4, 4.6, N), 17.1, 40.2),   # imc_materno
    np.clip(np.random.normal(13.4, 1.3, N), 9.8,  16.1),   # hb_materna_ajust
    np.clip(np.random.lognormal(np.log(18.7), 0.65, N), 3.2, 78.5),  # ferritina
    np.clip(np.random.normal(348.6, 98.2, N), 162.0, 710.0),  # acido_folico
    np.clip(np.random.normal(68.4, 12.8, N), 35.0, 115.0),  # zinc
    np.clip(np.random.normal(10.9, 3.6, N), 2.8, 21.4),     # ganancia_peso
    np.clip(np.random.normal(27.5, 4.2, N), 18.0, 35.0),    # edad_materna
    np.random.choice([0,1,2,3], N, p=[0.45,0.35,0.15,0.05]).astype(float),  # paridad
    np.clip(np.random.normal(25.7, 3.9, N), 17.8, 38.4),    # imc_paterno
    np.clip(np.random.normal(15.2, 1.5, N), 11.6, 18.3),    # hb_paterna_ajust
    np.clip(np.random.normal(29.8, 4.8, N), 18.0, 45.0),    # edad_paterna
    np.clip(np.random.normal(6.2,  1.8, N), 1.0,  10.0),    # divers_dietetica
])

score = (0.35*(X[:,5]-10.9)/3.6 + 0.28*(X[:,0]-26.4)/4.6 +
         0.18*(X[:,1]-13.4)/1.3 + 0.12*(X[:,2]-18.7)/14.8 +
         0.12*(X[:,8]-25.7)/3.9 + np.random.normal(0, 0.45, N))

p10 = np.percentile(score, 10.4)
p90 = np.percentile(score, 89.7)
y   = np.where(score < p10, 0, np.where(score > p90, 2, 1))

# ── Preprocesamiento ───────────────────────────────────────────────────────
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                        random_state=SEED, stratify=y)
prepro = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
])
Xtr_pp = prepro.fit_transform(Xtr)
Xbal, ybal = SMOTE(random_state=SEED).fit_resample(Xtr_pp, ytr)

print(f"\nDataset: {N} parejas | Train: {len(Xtr)} | Test: {len(Xte)}")
print(f"Post-SMOTE: {len(Xbal)} instancias")

# ── Entrenar los 5 modelos ─────────────────────────────────────────────────
modelos = {
    'Árbol de Decisión': DecisionTreeClassifier(
        max_depth=5, criterion='gini',
        min_samples_split=5, min_samples_leaf=2,
        random_state=SEED),
    'Regresión Logística': LogisticRegression(
        C=1.0, solver='lbfgs', penalty='l2',
        max_iter=1000, random_state=SEED),
    'SVM': SVC(
        C=10, kernel='rbf', gamma='scale',
        probability=True, random_state=SEED),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=10,
        max_features='sqrt', random_state=SEED),
    'XGBoost': XGBClassifier(
        learning_rate=0.1, n_estimators=200,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=SEED,
        eval_metric='mlogloss', verbosity=0),
}

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
Xte_pp = prepro.transform(Xte)

print(f"\n{'Algoritmo':22s} {'ACC':>6} {'F1':>6} {'AUC':>6}")
print("-" * 40)
for nombre, modelo in modelos.items():
    modelo.fit(Xbal, ybal)
    yp  = modelo.predict(Xte_pp)
    ypr = modelo.predict_proba(Xte_pp)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average='weighted', zero_division=0)
    auc = roc_auc_score(yte, ypr, multi_class='ovr')
    print(f"{nombre:22s} {acc:6.3f} {f1:6.3f} {auc:6.3f}")

# ── Guardar ────────────────────────────────────────────────────────────────
joblib.dump(modelos, 'modelos.pkl')
joblib.dump(prepro,  'preprocesador.pkl')

print("\n✅ modelos.pkl generado con tu versión local de XGBoost")
print("✅ preprocesador.pkl generado")
print("\nAhora ejecuta:  streamlit run app.py")
