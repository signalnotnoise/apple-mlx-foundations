import numpy as np

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"

def colored_print(text: str, color: str = CYAN) -> None:
    print(f"{color}{text}{RESET}")

def colored_metric(name: str, a: np.ndarray, b: np.ndarray) -> str:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    
    if max_abs < 1e-4:
        color = GREEN
    elif max_abs < 1e-2:
        color = YELLOW
    else:
        color = RED
    
    return f"{color}{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rmse={rmse:.6e}{RESET}"
