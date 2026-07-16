"""
DAE-PINN Digital Twin Interactive Predictor App Backend
======================================================
Serves API endpoints to perform real-time inferences and check physical constraint satisfaction.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Ensure workspace root is in path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Globals for model and scalers
MODEL = None
SCALERS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    global MODEL, SCALERS
    # Load model JIT TorchScript
    model_path = BASE_DIR / "phase5_dae_pinn" / "deployment" / "dae_pinn_model.pt"
    if not model_path.exists():
        # Fallback to checkers
        model_path = BASE_DIR / "phase5_dae_pinn" / "checkpoints" / "dae_pinn_best.pt"
        if not model_path.exists():
            raise FileNotFoundError("Model JIT file and checkpoints not found. Please run training first.")
    
    print(f"Loading model from: {model_path}")
    if model_path.suffix == '.pt' and 'best' not in model_path.name:
        MODEL = torch.jit.load(str(model_path), map_location=DEVICE)
    else:
        # It's a dict checkpoint, load state dict
        from phase5_dae_pinn.models.dae_pinn import DAEPINNModel
        ckpt = torch.load(model_path, map_location=DEVICE)
        config = ckpt['config']
        MODEL = DAEPINNModel(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['output_dim'],
            hidden_dims=config['model']['hidden_dims'],
            dropout_rate=config['model'].get('dropout_rate', 0.05),
        )
        MODEL.load_state_dict(ckpt['model_state'])
        MODEL.to(DEVICE)
    MODEL.eval()

    # Load scalers
    scalers_path = BASE_DIR / "phase4_pinn" / "datasets" / "normalized" / "scalers.pkl"
    if not scalers_path.exists():
        raise FileNotFoundError("Normalization scalers not found at phase4_pinn/datasets/normalized/scalers.pkl")
    
    print(f"Loading scalers from: {scalers_path}")
    with open(scalers_path, 'rb') as f:
        SCALERS = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        Vin = float(data.get('Vin', 48.0))
        D = float(data.get('D', 0.6))
        Fs = float(data.get('Fs', 50000.0))
        L = float(data.get('L', 50e-6))
        C = float(data.get('C', 47e-6))
        Rload = float(data.get('Rload', 5.0))
        t_end = float(data.get('t_end', 0.005))
        steps = int(data.get('steps', 200))

        # Build t evaluation array
        t_eval = np.linspace(0, t_end, steps, dtype=np.float32)

        # Build raw inputs: [t, Vin, D, Fs, L, C, Rload]
        raw_inputs = np.column_stack([
            t_eval,
            np.full(steps, Vin, dtype=np.float32),
            np.full(steps, D, dtype=np.float32),
            np.full(steps, Fs, dtype=np.float32),
            np.full(steps, L, dtype=np.float32),
            np.full(steps, C, dtype=np.float32),
            np.full(steps, Rload, dtype=np.float32),
        ])

        # Normalize using scaler_X
        norm_inputs = SCALERS['X'].transform(raw_inputs).astype(np.float32)
        X_tensor = torch.tensor(norm_inputs, device=DEVICE)

        # Predict
        with torch.no_grad():
            preds = MODEL(X_tensor).cpu().numpy() # [Vout, IL, Vc, dIL_dt, dVc_dt]

        Vout = preds[:, 0]
        IL = preds[:, 1]
        Vc = preds[:, 2]
        dIL_dt = preds[:, 3]
        dVc_dt = preds[:, 4]

        # Evaluate DAE residuals
        # KVL residual: dIL/dt - (Vin - (1 - D) * Vout) / L
        # KCL residual: dVc/dt - ((1 - D) * IL - Vc / Rload) / C
        # Algebraic residual: Vout - Vc
        res_kvl = dIL_dt - (Vin - (1.0 - D) * Vout) / L
        res_kcl = dVc_dt - ((1.0 - D) * IL - Vc / Rload) / C
        res_alg = Vout - Vc

        # Compute power in / power out to measure conservation
        # Pin = Vin * IL * D
        # Pout = Vout^2 / Rload
        # Efficiency = Pout / Pin
        Pin = Vin * IL * D
        Pout = Vout**2 / Rload
        
        # Calculate theoretical steady-state
        Vout_steady = Vin / (1.0 - D)
        IL_steady = Vin / (((1.0 - D)**2) * Rload)

        # Prepare outputs
        response_data = {
            't': t_eval.tolist(),
            'Vout': Vout.tolist(),
            'IL': IL.tolist(),
            'Vc': Vc.tolist(),
            'dIL_dt': dIL_dt.tolist(),
            'dVc_dt': dVc_dt.tolist(),
            'residuals': {
                'kvl': res_kvl.tolist(),
                'kcl': res_kcl.tolist(),
                'alg': res_alg.tolist(),
                'kvl_mean_abs': float(np.mean(np.abs(res_kvl))),
                'kcl_mean_abs': float(np.mean(np.abs(res_kcl))),
                'alg_mean_abs': float(np.mean(np.abs(res_alg)))
            },
            'power': {
                'Pin': Pin.tolist(),
                'Pout': Pout.tolist(),
                'Pin_avg': float(np.mean(Pin)),
                'Pout_avg': float(np.mean(Pout)),
                'efficiency': float(np.clip(np.mean(Pout) / (np.mean(Pin) + 1e-9) * 100.0, 0, 100))
            },
            'steady_state': {
                'Vout': Vout_steady,
                'IL': IL_steady
            }
        }

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Digital Twin Predictor server")
    parser.add_argument('--port', type=int, default=5000, help='Port to run Flask server (default 5000)')
    args = parser.parse_args()

    # Load resources before starting
    load_resources()

    app.run(host="0.0.0.0", port=args.port, debug=False)
