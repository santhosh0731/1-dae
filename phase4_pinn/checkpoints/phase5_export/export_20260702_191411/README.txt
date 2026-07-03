============================================================
  PHASE 4 EXPORT — Ready for Phase 5 DAE-PINN
============================================================

  Export timestamp: 20260702_191411

  FILES
  -----
  pinn_weights.pt           Load with torch.load()
  normalization_scalers.pkl Load with pickle.load()
  physics_residuals.npz     Load with np.load()
  phase4_metrics.json       Human-readable benchmark
  pinn_config.yaml          Hyperparameters
  dae_formulation.py        F(x,dx/dt,z)=0 equations
  solver_recommendation.txt Best solver for IRK

  LOADING IN PHASE 5
  ------------------
  import torch, pickle
  from phase4_pinn.models.pinn_model import PINNModel

  ckpt = torch.load('pinn_weights.pt')
  model = PINNModel(input_dim=7, output_dim=5)
  model.load_state_dict(ckpt['model_state'])

  with open('normalization_scalers.pkl','rb') as f:
      scalers = pickle.load(f)

  # Phase 5: extend model with IRK layer and DAE constraints
  # See dae_formulation.py for F(x,dx/dt,z)=0
