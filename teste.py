import sys
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Caminho do projeto (sem caminho absoluto)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantumnet.topology import Network

# ============================================================
# Configurações visuais para os gráficos do artigo
# ============================================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# ============================================================
# Diretório de saída
# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Sementes reproduzíveis
# ============================================================
POLICY_SEED = {
    "threshold": 11,
    "on_demand": 22,
    "hybrid": 33,
}

SCENARIO_SEED = {
    "baseline": 100,
    "moderada": 200,
    "sobrecarga": 300,
    "burst": 400,
    "threshold_low": 500,
    "threshold_mid": 600,
    "threshold_high": 700,
}


def make_seed(policy: str, scenario: str, trial: int, extra: int = 0) -> int:
    """Combina seeds fixas em um inteiro determinístico para RNG."""
    if policy not in POLICY_SEED:
        raise KeyError(f"Policy desconhecida: {policy}")
    if scenario not in SCENARIO_SEED:
        raise KeyError(f"Cenário desconhecido: {scenario}")

    return int(
        POLICY_SEED[policy] * 1_000_000
        + SCENARIO_SEED[scenario] * 1_000
        + int(trial)
        + int(extra)
    )


def get_rng(policy: str, scenario: str, trial: int, extra: int = 0) -> random.Random:
    return random.Random(make_seed(policy, scenario, trial, extra=extra))


def seed_numpy(policy: str, scenario: str, trial: int, extra: int = 0) -> None:
    np.random.seed(make_seed(policy, scenario, trial, extra=extra) % (2**32 - 1))


def save_summary(df: pd.DataFrame, group_cols: list[str], exp_name: str) -> None:
    metric_cols = [
        "service_rate",
        "denied_rate",
        "avg_buffer_occupancy_bits",
        "replenishment_events",
    ]

    summary = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_path = OUTPUT_DIR / f"{exp_name}_summary.csv"
    summary.to_csv(summary_path)
    print(f"Resumo salvo em: {summary_path}")


def plot_and_save(df: pd.DataFrame, x_col: str, hue_col: str | None, exp_name: str) -> None:
    metrics = {
        "service_rate": "Taxa de Atendimento",
        "denied_rate": "Taxa de Negação",
        "avg_buffer_occupancy_bits": "Ocupação Média do Buffer (bits)",
        "replenishment_events": "Eventos de Reposição",
    }

    for metric, ylabel in metrics.items():
        plt.figure()

        if hue_col:
            sns.barplot(
                data=df,
                x=x_col,
                y=metric,
                hue=hue_col,
                capsize=0.1,
                errorbar="sd",
            )
        else:
            sns.barplot(
                data=df,
                x=x_col,
                y=metric,
                capsize=0.1,
                errorbar="sd",
                color="skyblue",
            )

        plt.title(f"{exp_name} - {ylabel}")
        plt.ylabel(ylabel)
        plt.tight_layout()

        fig_path = OUTPUT_DIR / f"{exp_name}_{metric}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

    csv_filename = OUTPUT_DIR / f"{exp_name}_resultados.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Resultados guardados em: {csv_filename}")


# ============================================================
# EXPERIMENTO 1: Comparação básica entre políticas
# ============================================================
def run_exp1_trial(args):
    policy, trial = args
    rng = get_rng(policy, "baseline", trial)
    seed_numpy(policy, "baseline", trial)

    net = Network()
    net.set_ready_topology("Linha", 4)
    net.controller.set_policy(policy)

    link_pairs = [(0, 1), (1, 2), (2, 3)]
    minimum_stock_bits = 128
    requests_per_link = 15
    request_bit_options = [32, 48, 64]

    for a, b in link_pairs:
        net.controller.set_minimum_stock(a, b, minimum_stock_bits)

    buffer_history = []

    for a, b in link_pairs:
        for _ in range(requests_per_link):
            req_bits = rng.choice(request_bit_options)
            net.controller.handle_key_request(a, b, req_bits)
            buffer_history.append(int(net.get_qkd_link_state(a, b)["bits_available"]))

    served, denied, replenishments = 0, 0, 0
    for a, b in link_pairs:
        state = net.get_qkd_link_state(a, b)
        served += int(state["served_requests"])
        denied += int(state["denied_requests"])
        replenishments += int(state["replenishment_events"])

    total_reqs = served + denied
    return {
        "policy": policy,
        "trial": trial,
        "service_rate": served / total_reqs if total_reqs > 0 else 0,
        "denied_rate": denied / total_reqs if total_reqs > 0 else 0,
        "replenishment_events": replenishments,
        "avg_buffer_occupancy_bits": float(np.mean(buffer_history)) if buffer_history else 0,
    }


def run_experiment_1(max_workers: int = 10, trials: int = 30) -> pd.DataFrame:
    policies = ["threshold", "on_demand", "hybrid"]
    tasks = [(p, t) for p in policies for t in range(trials)]

    print(f"A iniciar {len(tasks)} tarefas do Experimento 1...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_exp1_trial, tasks))

    df_exp1 = pd.DataFrame(results)
    plot_and_save(df_exp1, x_col="policy", hue_col=None, exp_name="Exp1_Baseline")
    save_summary(df_exp1, ["policy"], "Exp1_Baseline")

    print("\nResumo Experimento 1:")
    print(df_exp1.groupby("policy").agg(["mean", "std"]).round(3))
    return df_exp1


# ============================================================
# EXPERIMENTO 2: Carga moderada, sobrecarga e burst
# ============================================================
def run_exp2_trial(args):
    policy, load_name, pattern_type, trial = args
    rng = get_rng(policy, pattern_type, trial)
    seed_numpy(policy, pattern_type, trial)

    net = Network()
    net.set_ready_topology("Linha", 4)
    net.controller.set_policy(policy)

    link_pairs = [(0, 1), (1, 2), (2, 3)]
    minimum_stock_bits = 128
    requests_per_link = 20

    for a, b in link_pairs:
        net.controller.set_minimum_stock(a, b, minimum_stock_bits)

    buffer_history = []

    for a, b in link_pairs:
        for _ in range(requests_per_link):
            if pattern_type == "moderada":
                req_bits = rng.choice([32, 48, 64])
            elif pattern_type == "sobrecarga":
                req_bits = rng.choice([128, 192, 256])
            elif pattern_type == "burst":
                req_bits = 512 if rng.random() > 0.8 else 16
            else:
                raise ValueError(f"pattern_type inválido: {pattern_type}")

            net.controller.handle_key_request(a, b, req_bits)
            buffer_history.append(int(net.get_qkd_link_state(a, b)["bits_available"]))

    served, denied, replenishments = 0, 0, 0
    for a, b in link_pairs:
        state = net.get_qkd_link_state(a, b)
        served += int(state["served_requests"])
        denied += int(state["denied_requests"])
        replenishments += int(state["replenishment_events"])

    total_reqs = served + denied
    return {
        "policy": policy,
        "load": load_name,
        "trial": trial,
        "service_rate": served / total_reqs if total_reqs > 0 else 0,
        "denied_rate": denied / total_reqs if total_reqs > 0 else 0,
        "replenishment_events": replenishments,
        "avg_buffer_occupancy_bits": float(np.mean(buffer_history)) if buffer_history else 0,
    }


def run_experiment_2(max_workers: int = 10, trials: int = 30) -> pd.DataFrame:
    policies = ["threshold", "on_demand", "hybrid"]
    scenarios = [
        ("Moderada", "moderada"),
        ("Sobrecarga", "sobrecarga"),
        ("Rajada (Burst)", "burst"),
    ]

    tasks = [(p, l_name, p_type, t) for p in policies for l_name, p_type in scenarios for t in range(trials)]

    print(f"A iniciar {len(tasks)} tarefas do Experimento 2...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_exp2_trial, tasks))

    df_exp2 = pd.DataFrame(results)
    df_exp2["load"] = pd.Categorical(
        df_exp2["load"],
        categories=["Moderada", "Sobrecarga", "Rajada (Burst)"],
        ordered=True,
    )

    plot_and_save(df_exp2, x_col="load", hue_col="policy", exp_name="Exp2_Carga_Burst")
    save_summary(df_exp2, ["policy", "load"], "Exp2_Carga_Burst")

    print("\nResumo Experimento 2:")
    print(df_exp2.groupby(["policy", "load"], observed=False).agg(["mean", "std"]).round(3))
    return df_exp2


# ============================================================
# EXPERIMENTO 3: Sensibilidade ao limiar
# ============================================================
def run_exp3_trial(args):
    policy, threshold, trial = args

    rng = get_rng(policy, "threshold_mid", trial, extra=int(threshold))
    seed_numpy(policy, "threshold_mid", trial, extra=int(threshold))

    net = Network()
    net.set_ready_topology("Linha", 4)
    net.controller.set_policy(policy)

    link_pairs = [(0, 1), (1, 2), (2, 3)]
    requests_per_link = 15
    request_bit_options = [48, 64, 96]

    for a, b in link_pairs:
        net.controller.set_minimum_stock(a, b, threshold)

    buffer_history = []

    for a, b in link_pairs:
        for _ in range(requests_per_link):
            req_bits = rng.choice(request_bit_options)
            net.controller.handle_key_request(a, b, req_bits)
            buffer_history.append(int(net.get_qkd_link_state(a, b)["bits_available"]))

    served, denied, replenishments = 0, 0, 0
    for a, b in link_pairs:
        state = net.get_qkd_link_state(a, b)
        served += int(state["served_requests"])
        denied += int(state["denied_requests"])
        replenishments += int(state["replenishment_events"])

    total_reqs = served + denied
    return {
        "policy": policy,
        "threshold": threshold,
        "trial": trial,
        "service_rate": served / total_reqs if total_reqs > 0 else 0,
        "denied_rate": denied / total_reqs if total_reqs > 0 else 0,
        "replenishment_events": replenishments,
        "avg_buffer_occupancy_bits": float(np.mean(buffer_history)) if buffer_history else 0,
    }


def run_experiment_3(max_workers: int = 10, trials: int = 30) -> pd.DataFrame:
    sensitive_policies = ["threshold", "hybrid"]
    threshold_values = [32, 64, 128, 256, 512]

    tasks = [(p, th, t) for p in sensitive_policies for th in threshold_values for t in range(trials)]

    print(f"A iniciar {len(tasks)} tarefas do Experimento 3...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_exp3_trial, tasks))

    df_exp3 = pd.DataFrame(results)
    plot_and_save(df_exp3, x_col="threshold", hue_col="policy", exp_name="Exp3_Limiares")
    save_summary(df_exp3, ["policy", "threshold"], "Exp3_Limiares")

    print("\nResumo Experimento 3:")
    print(df_exp3.groupby(["policy", "threshold"]).agg(["mean", "std"]).round(3))
    return df_exp3


def main() -> None:
    run_experiment_1(max_workers=10, trials=2)
    run_experiment_2(max_workers=10, trials=2)
    run_experiment_3(max_workers=10, trials=2)


if __name__ == "__main__":
    main()