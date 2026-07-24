# NutriML-Pareja — App Streamlit (configuración final: 22 variables, 6 salidas)
# Prueba de concepto: los modelos se entrenan sobre datos SIMULADOS fundamentados
# en la literatura (semilla 42). NO usar para decisiones clínicas.
import numpy as np, pandas as pd, streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="NutriML-Pareja", page_icon="🧬", layout="wide")

FEATS = ["edad_m","paridad","imc_m","gp","hb_m","ferritina","glu_m","hba1c_m","folico_sup","talla_m",
         "ant_ptr","pa_alta","infec","edad_p","imc_p","hb_p","glu_p","diet_p","conc_p","mot_p","morf_p","talla_p"]
LABELS = {"y_peso":["Bajo peso","Adecuado","Macrosomía"], "y_eg":["Pretérmino","A término"],
          "y_pesoEG":["PEG","AEG","GEG"], "y_talla":["Talla baja","Adecuada","Talla alta"],
          "y_apgar":["Deprimido","Vigoroso"], "y_viab":["Pérdida / no logro","Viable"]}
TITLES = {"y_peso":"Peso al nacer","y_eg":"Edad gestacional","y_pesoEG":"Peso para la edad gestacional",
          "y_talla":"Talla al nacer","y_apgar":"APGAR","y_viab":"Viabilidad / logro de embarazo"}
NORMAL_IDX = {"y_peso":1,"y_eg":1,"y_pesoEG":1,"y_talla":1,"y_apgar":1,"y_viab":1}

@st.cache_resource(show_spinner="Entrenando modelos (una sola vez)…")
def train():
    SEED=42; rng=np.random.default_rng(SEED); np.random.seed(SEED); N=470
    z=lambda a:(a-np.nanmean(a))/np.nanstd(a)
    edad_m=np.clip(rng.normal(27,4.5,N),18,35); edad_p=np.clip(edad_m+rng.normal(2.5,3,N),18,40)
    paridad=rng.poisson(1.1,N).clip(0,5); imc_m=np.clip(rng.normal(26.4,4.6,N),16.5,41); imc_p=np.clip(rng.normal(25.7,3.9,N),17,39)
    hb_m=np.clip(rng.normal(13.4,1.3,N),9,16.5); hb_p=np.clip(rng.normal(15.2,1.5,N),11,18.5)
    ferritina=np.clip(rng.gamma(2.4,9.5,N),3,90); gp=np.clip(rng.gamma(9,1.2,N),2.5,22)
    diet_p=np.clip(rng.normal(6,1.8,N),1,10); folico_sup=(rng.random(N)<0.6).astype(int)
    glu_m=np.clip(rng.normal(86,9,N)+(rng.random(N)<0.1)*rng.normal(38,12,N),60,220)
    hba1c_m=np.clip(2.6+0.031*glu_m+rng.normal(0,0.25,N),4.2,10.5)
    glu_p=np.clip(rng.normal(88,10,N)+(rng.random(N)<0.1)*rng.normal(35,12,N),60,220)
    conc_p=np.clip(rng.normal(45,25,N),2,150); mot_p=np.clip(rng.normal(48,14,N),5,85); morf_p=np.clip(rng.normal(6,2.2,N),0,18)
    spz=((conc_p-45)/25+(mot_p-48)/14+(morf_p-6)/2.2)/3
    talla_m=np.clip(rng.normal(152,6,N),138,172); talla_p=np.clip(rng.normal(163,7,N),146,186)
    ant_ptr=(rng.random(N)<0.15).astype(int); pa_alta=(rng.random(N)<0.11).astype(int); infec=(rng.random(N)<0.20).astype(int)
    z_gp=z(gp); z_imcm=z(imc_m); z_hbm=(hb_m-13.4)/1.3; z_ferr=z(ferritina); z_imcp=z(imc_p); z_glu=z(glu_m); z_tm=z(talla_m); z_tp=z(talla_p)
    vlogit=1.9+2.4*spz-0.6*((edad_m>35).astype(float))-0.5*z(hba1c_m)-0.4*((imc_m>=30).astype(float))-0.3*z(edad_p)
    y_viab=(rng.random(N)<1/(1+np.exp(-vlogit))).astype(int)
    eg=np.clip(39.2+0.35*z_gp+0.15*z_hbm-0.2*np.abs(z_imcm)+0.12*folico_sup+0.10*spz-1.9*ant_ptr-1.3*infec-1.1*pa_alta+rng.normal(0,1.0,N),30,41.5)
    peso=np.clip(3250+210*z_gp+150*z_imcm+120*z_hbm+80*z_ferr+70*z_imcp+120*z_glu+30*spz-140*pa_alta+30*z_tm+18*z_tp+195*(eg-38.8)+rng.normal(0,290,N),1400,4750)
    talla=np.clip(48+0.0022*(peso-3200)+0.8*(eg-38.8)+0.9*z_tm+0.7*z_tp+rng.normal(0,1.2,N),40,56)
    apgar1=np.clip(np.round(9.0+0.5*(eg-38.8)+0.0008*(peso-3200)-0.4*pa_alta+rng.normal(0,1.3,N)),1,10)
    df=pd.DataFrame(dict(edad_m=edad_m,paridad=paridad,imc_m=imc_m,gp=gp,hb_m=hb_m,ferritina=ferritina,glu_m=glu_m,hba1c_m=hba1c_m,
     folico_sup=folico_sup,talla_m=talla_m,ant_ptr=ant_ptr,pa_alta=pa_alta,infec=infec,edad_p=edad_p,imc_p=imc_p,hb_p=hb_p,glu_p=glu_p,
     diet_p=diet_p,conc_p=conc_p,mot_p=mot_p,morf_p=morf_p,talla_p=talla_p,peso=peso,eg=eg,talla=talla,apgar1=apgar1,y_viab=y_viab))
    df=df.drop(index=rng.choice(N,N-424,replace=False)).reset_index(drop=True)
    df["y_peso"]=df.peso.apply(lambda g:0 if g<2500 else(2 if g>=4000 else 1))
    df["y_eg"]=df.eg.apply(lambda s:0 if s<37 else 1)
    df["y_talla"]=df.talla.apply(lambda t:0 if t<47 else(2 if t>=53 else 1))
    df["y_apgar"]=df.apgar1.apply(lambda a:0 if a<=6 else 1)
    bb=np.polyfit(df.eg,df.peso,1); res=df.peso-np.polyval(bb,df.eg); p10,p90=np.percentile(res,[10,90])
    df["y_pesoEG"]=res.apply(lambda r:0 if r<p10 else(2 if r>p90 else 1))
    models={}
    for col in LABELS:
        models[col]=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
            ("clf",RandomForestClassifier(n_estimators=250,max_depth=10,class_weight="balanced",random_state=SEED))]).fit(df[FEATS],df[col].values)
    return models

models = train()

st.title("🧬 NutriML-Pareja")
st.caption("Sistema predictivo con Machine Learning para la evaluación nutricional y clínica preconcepcional de la pareja — Juliaca, altiplano peruano.")
st.warning("⚠️ Prueba de concepto entrenada con **datos simulados** fundamentados en la literatura. No debe usarse para decisiones clínicas hasta su validación con datos reales.")

with st.form("perfil"):
    cM, cP = st.columns(2)
    with cM:
        st.subheader("👩 Madre")
        edad_m=st.number_input("Edad (años)",18,45,27)
        paridad=st.number_input("Paridad (embarazos previos)",0,8,1)
        imc_m=st.number_input("IMC preconcepcional (kg/m²)",15.0,45.0,24.0,0.1)
        gp=st.number_input("Ganancia de peso pregestacional (kg)",0.0,30.0,11.0,0.1)
        hb_m=st.number_input("Hemoglobina AJUSTADA por altitud (g/dL)",8.0,18.0,13.5,0.1)
        ferritina=st.number_input("Ferritina (µg/L)",3.0,150.0,25.0,0.5)
        glu_m=st.number_input("Glucosa en ayunas (mg/dL)",60,260,85)
        hba1c_m=st.number_input("HbA1c (%)",4.0,12.0,5.2,0.1)
        folico_sup=1 if st.selectbox("Suplementación de ácido fólico",["Sí","No"])=="Sí" else 0
        talla_m=st.number_input("Talla materna (cm)",135,185,152)
        ant_ptr=1 if st.selectbox("Antecedente de parto pretérmino / aborto",["No","Sí"])=="Sí" else 0
        pa_alta=1 if st.selectbox("Presión arterial",["Normal","Elevada / HTA"])=="Elevada / HTA" else 0
        infec=1 if st.selectbox("Tamizaje de infección (ITU / vaginosis)",["Negativo","Positivo"])=="Positivo" else 0
    with cP:
        st.subheader("👨 Padre")
        edad_p=st.number_input("Edad (años) ",18,60,30)
        imc_p=st.number_input("IMC (kg/m²)",15.0,45.0,25.0,0.1)
        hb_p=st.number_input("Hemoglobina AJUSTADA por altitud (g/dL) ",10.0,20.0,15.2,0.1)
        glu_p=st.number_input("Glucosa en ayunas (mg/dL) ",60,260,88)
        diet_p=st.number_input("Diversidad dietética (n.º grupos)",0,10,6)
        conc_p=st.number_input("Concentración espermática (mill/mL)",0.0,200.0,45.0,1.0)
        mot_p=st.number_input("Motilidad progresiva (%)",0.0,100.0,48.0,1.0)
        morf_p=st.number_input("Morfología normal (%)",0.0,25.0,6.0,0.1)
        talla_p=st.number_input("Talla paterna (cm)",145,205,165)
    enviar=st.form_submit_button("🔮 Predecir salud de la progenie", use_container_width=True)

if enviar:
    row=pd.DataFrame([[edad_m,paridad,imc_m,gp,hb_m,ferritina,glu_m,hba1c_m,folico_sup,talla_m,ant_ptr,pa_alta,infec,
        edad_p,imc_p,hb_p,glu_p,diet_p,conc_p,mot_p,morf_p,talla_p]],columns=FEATS)
    st.subheader("📋 Panel de riesgo del neonato")
    cols=st.columns(3)
    for i,col in enumerate(["y_peso","y_eg","y_pesoEG","y_talla","y_apgar","y_viab"]):
        proba=models[col].predict_proba(row)[0]; classes=models[col].named_steps["clf"].classes_
        p={LABELS[col][c]:float(proba[j]) for j,c in enumerate(classes)}
        pred=max(p,key=p.get)
        p_normal=p.get(LABELS[col][NORMAL_IDX[col]],0.0); risk=1-p_normal
        color="🟢" if risk<0.15 else ("🟡" if risk<0.35 else "🔴")
        with cols[i%3]:
            st.markdown(f"**{color} {TITLES[col]}**")
            st.markdown(f"Predicción: **{pred}**")
            for k,v in sorted(p.items(),key=lambda x:-x[1]):
                st.progress(min(v,1.0), text=f"{k}: {v*100:.1f}%")
            st.divider()
    st.info("La hemoglobina debe ingresarse ya **ajustada por altitud** (factor OMS/OPS ≈ −3 g/dL para 3 824 m). "
            "El APGAR es una salida secundaria (depende de factores intraparto). Semáforo: 🟢 riesgo bajo · 🟡 moderado · 🔴 alto.")

st.caption("NutriML-Pareja · Universidad Nacional de Juliaca · Escuela Profesional de Ingeniería de Software y Sistemas · 2026")
