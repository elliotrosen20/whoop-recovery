import gradio as gr
import pandas as pd
import xgboost as xgb
import joblib

model = joblib.load("xgboost_model_personal.joblib")

def predict(
    rhr, hrv, temp, spo2, resp,
    asleep, in_bed, light, deep, rem, awake,
    sleep_need, sleep_debt
):
    sleep_efficiency = (asleep / in_bed) * 100 if in_bed else 0
    sleep_performance = ((sleep_need - sleep_debt) / sleep_need) * 100 if sleep_need else 0
    light_ratio = light / asleep if asleep else 0
    deep_ratio = deep / asleep if asleep else 0
    rem_ratio = rem / asleep if asleep else 0

    data = {
        "Resting heart rate (bpm)": rhr,
        "Heart rate variability (ms)": hrv,
        "Skin temp (celsius)": temp,
        "Blood oxygen %": spo2,
        "Respiratory rate (rpm)": resp,
        "Asleep duration (min)": asleep,
        "In bed duration (min)": in_bed,
        "Light sleep duration (min)": light,
        "Deep (SWS) duration (min)": deep,
        "REM duration (min)": rem,
        "Awake duration (min)": awake,
        "Sleep need (min)": sleep_need,
        "Sleep debt (min)": sleep_debt,
        "Sleep efficiency %": sleep_efficiency,
        "Sleep performance %": sleep_performance,
        "Light sleep ratio": light_ratio,
        "Deep sleep ratio": deep_ratio,
        "REM sleep ratio": rem_ratio
    }

    expected_features = [
      'Resting heart rate (bpm)', 'Heart rate variability (ms)',
      'Skin temp (celsius)', 'Blood oxygen %', 'Sleep performance %',
      'Respiratory rate (rpm)', 'Asleep duration (min)',
      'In bed duration (min)', 'Light sleep duration (min)',
      'Deep (SWS) duration (min)', 'REM duration (min)',
      'Awake duration (min)', 'Sleep need (min)', 'Sleep debt (min)',
      'Sleep efficiency %', 'Deep sleep ratio', 'REM sleep ratio',
      'Light sleep ratio'
    ]

    df = pd.DataFrame([data])
    df = df[expected_features]

    pred = model.predict(df)[0]
    return round(pred, 2)

inputs = [
    gr.Slider(30, 100, label="Resting heart rate (bpm)"),
    gr.Slider(10, 200, label="Heart rate variability (ms)"),
    gr.Slider(30, 40, label="Skin temp (°C)"),
    gr.Slider(85, 100, label="Blood oxygen %"),
    gr.Slider(10, 25, label="Respiratory rate (rpm)"),
    gr.Slider(200, 600, label="Asleep duration (min)"),
    gr.Slider(200, 700, label="In bed duration (min)"),
    gr.Slider(0, 400, label="Light sleep duration (min)"),
    gr.Slider(0, 200, label="Deep (SWS) duration (min)"),
    gr.Slider(0, 200, label="REM duration (min)"),
    gr.Slider(0, 200, label="Awake duration (min)"),
    gr.Slider(300, 600, label="Sleep need (min)"),
    gr.Slider(0, 200, label="Sleep debt (min)")
]

output = gr.Number(label="Predicted WHOOP Recovery Score")

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="WHOOP Recovery Score Predictor",
    description="Adjust the sliders to simulate different biometric states and estimate recovery score.",
)

app.launch()