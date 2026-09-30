import os
import argparse
import json
import joblib
import numpy as np
import torch
import importlib.util
from torch.utils.data import TensorDataset, Subset
from dataset_inference import DatasetInferenceDefense
import time


def _load_config_factory(input_dir):
    """Best-effort resolver for config() used to rebuild the model architecture."""
    candidate_paths = [
        "/app/config_model.py",
        os.path.join(input_dir, "config_model.py"),
        "/input/config_model.py",
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("config_model", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, "config", None)
        if callable(factory):
            print(f"Loaded model config from: {path}")
            return factory
    return None


def load_model(model_path, device='cpu'):
    """Load model with config_model.py when available, else fallback to full model."""
    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        print("Loaded serialized torch.nn.Module directly from checkpoint")
        return model

    input_dir = os.path.dirname(model_path)
    config_factory = _load_config_factory(input_dir)
    if config_factory is None:
        available = ", ".join(sorted(os.listdir(input_dir))) if os.path.isdir(input_dir) else "<missing input dir>"
        raise FileNotFoundError(
            "Unable to load model architecture: checkpoint is not a serialized torch.nn.Module "
            "and config_model.py was not found in /app, /input, or input directory. "
            f"Input dir contents: {available}"
        )

    model = config_factory()
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device)
    return model


def load_numpy_dataset(data_path, labels_path):
    data = np.load(data_path)  # shape: (N, C, H, W)
    labels = np.load(labels_path)  # shape: (N,)
    tensor_data = torch.tensor(data, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(tensor_data, tensor_labels)


def main(args):
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output, 'dataset_inference_log.txt')
    with open(log_file, 'w') as f:
        f.write("=== Dataset Inference Defense Log ===\n")
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    with open(log_file, 'a') as f:
        f.write(f"Using device: {device}\n")

    victim_model_path = os.path.join(args.input_dir, 'model.pt')
    victim_model = load_model(victim_model_path, device)
    
    with open(log_file, 'a') as f:
        f.write(f"Model loaded successfully from {victim_model_path}\n")

    # Load datasets
    private_dataset = load_numpy_dataset(
        os.path.join(args.input_dir, 'data.npy'),
        os.path.join(args.input_dir, 'labels.npy')
    )
    public_dataset = load_numpy_dataset(
        os.path.join(args.input_dir, 'test_data.npy'),
        os.path.join(args.input_dir, 'test_labels.npy')
    )

    # Use subset for efficiency
    private_subset = Subset(private_dataset, range(min(100, len(private_dataset))))
    public_subset = Subset(public_dataset, range(min(100, len(public_dataset))))
    
    with open(log_file, 'a') as f:
        f.write(f"Private dataset size: {len(private_subset)}\n")
        f.write(f"Public dataset size: {len(public_subset)}\n\n")

    # Initialize dataset inference tool
    tool = DatasetInferenceDefense(device=device)
    
    with open(log_file, 'a') as f:
        f.write("Training regressor to distinguish private vs public embeddings...\n")
    
    # Train regressor
    regressor = tool.train_regressor(victim_model, private_subset, public_subset, num_points=len(private_subset))
    
    # Save regressor
    regressor_path = os.path.join(args.output, 'regressor.joblib')
    joblib.dump(regressor, regressor_path)
    
    with open(log_file, 'a') as f:
        f.write(f"Regressor saved to {regressor_path}\n")

    # Run inference
    with open(log_file, 'a') as f:
        f.write("Running dataset inference analysis...\n")
    
    scores = tool.run_inference(victim_model, private_subset, public_subset, num_points=len(private_subset))
    
    # Save detailed results
    results_path = os.path.join(args.output, 'dataset_inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(scores, f, indent=2)
    
    # Log results
    with open(log_file, 'a') as f:
        f.write(f"\n=== DATASET INFERENCE RESULTS ===\n")
        f.write(f"Mean private score: {scores['mean_private_score']:.6f}\n")
        f.write(f"Mean public score: {scores['mean_public_score']:.6f}\n")
        f.write(f"Difference: {scores['mean_private_score'] - scores['mean_public_score']:.6f}\n")
        f.write(f"P-value: {scores['p_value']:.6e}\n")
        f.write(f"Decision: {scores['inference_result']}\n")
        #f.write(f"Alpha threshold: {scores['alpha_threshold']}\n")
        if 'detailed_scores' in scores:
            f.write(f"Private scores std: {np.std(scores['detailed_scores']['private_scores']):.6f}\n")
            f.write(f"Public scores std: {np.std(scores['detailed_scores']['public_scores']):.6f}\n")
        f.write(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Dataset Inference completed. Results saved to: {args.output}")
    print(f"Inference result: {scores['inference_result']}")
    print(f"P-value: {scores['p_value']:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="/data", help='Directory containing model.pt and .npy files')
    parser.add_argument('--output', type=str, default="/output", required=True, help='Directory to save regressor and scores')
    args = parser.parse_args()
    main(args)