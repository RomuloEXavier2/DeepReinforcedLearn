# train_ppo.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baseline_3.common.callbacks import CheckpointCallback, EvalCallback
from environment import PalletizingEnv

# --- Configurações do Treinamento ---
TIMESTEPS = 1_000_000
N_ENVS = 4  # Número de ambientes para treinar em paralelo
EVAL_FREQ = 25000 // N_ENVS
CHECKPOINT_FREQ = 50000 // N_ENVS

# --- Estrutura de Pastas ---
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
current_dir = Path(__file__).parent.absolute()
SAVE_DIR = current_dir / "models" / f"PPO_{run_timestamp}"


def plot_evaluation_results(log_path):
    """
    Carrega os resultados da avaliação do EvalCallback e plota o gráfico de recompensa.
    """
    try:
        data = np.load(os.path.join(log_path, "evaluations.npz"))
        timesteps = data['timesteps']
        results = data['results']

        mean_reward = np.mean(results, axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, mean_reward)
        plt.title('Curva de Aprendizado (Recompensa Média por Avaliação)')
        plt.xlabel('Timesteps')
        plt.ylabel('Recompensa Média')
        plt.grid(True)
        plt.show()
    except FileNotFoundError:
        print("\nArquivo 'evaluations.npz' não encontrado. Gráfico não gerado.")
        print("Certifique-se de que o treinamento rodou por tempo suficiente para uma avaliação ocorrer.")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 Modelos e logs serão salvos em: {SAVE_DIR}")

    # --- Ambiente ---
    env_kwargs = {'pallet_size': (5, 5), 'max_height': 5, 'lookahead_k': 3}
    train_env = make_vec_env(PalletizingEnv, n_envs=N_ENVS, env_kwargs=env_kwargs)
    eval_env = PalletizingEnv(**env_kwargs)

    # --- Callbacks para um Treinamento Robusto ---
    # Salva o melhor modelo com base no desempenho da avaliação
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR),
        log_path=str(SAVE_DIR),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )

    # --- Modelo PPO ---
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=None,  # <-- TENSORBOARD DESABILITADO PARA EVITAR O ERRO
        gamma=0.95,
        device="cpu"
    )

    print(f"--- 🚀 Iniciando Treinamento com PPO no dispositivo: {model.device} ---")
    start_time = time.time()

    try:
        # O treinamento agora usa apenas o EvalCallback
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=eval_callback
        )
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        raise

    end_time = time.time()
    print(f"--- ✅ Treinamento Concluído em {(end_time - start_time) / 60:.2f} minutos ---")

    # O melhor modelo já foi salvo pelo callback, mas podemos salvar o final também.
    model.save(str(SAVE_DIR / "final_model"))
    print(f"💾 Modelo Final salvo em: {SAVE_DIR / 'final_model.zip'}")
    print(f"🏆 Melhor Modelo salvo durante o treino em: {SAVE_DIR / 'best_model.zip'}")

    # Plota os resultados da avaliação no final
    plot_evaluation_results(log_path=str(SAVE_DIR))