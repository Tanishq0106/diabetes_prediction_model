# ── Cell 17: Save model and scaler ──────────────────
import joblib

joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
print("Model and scaler saved!")