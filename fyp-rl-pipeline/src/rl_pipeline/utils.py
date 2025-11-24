def log(message):
    print(f"[LOG] {message}")

def calculate_metrics(rewards):
    return {
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
    }

def save_model(model, filepath):
    import torch
    torch.save(model.state_dict(), filepath)
    log(f"Model saved to {filepath}")

def load_model(model, filepath):
    import torch
    model.load_state_dict(torch.load(filepath))
    log(f"Model loaded from {filepath}")