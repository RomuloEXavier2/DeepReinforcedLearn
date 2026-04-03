# run_project.py

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import PalletizingEnv

def train_new_agent(timesteps, save_path):
    """Função para treinar um novo agente PPO do zero."""
    print("--- 🚀 Iniciando Novo Treinamento ---")

    env_kwargs = {'pallet_size': (5, 5), 'max_height': 5, 'lookahead_k': 3}
    train_env = make_vec_env(PalletizingEnv, n_envs=4, env_kwargs=env_kwargs)

    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu", gamma=0.95)
    model.learn(total_timesteps=timesteps)

    model.save(save_path)
    print(f"\n--- ✅ Treinamento Concluído. Modelo salvo em: {save_path}.zip ---")


def evaluate_model_visual(model_path, items):
    """Carrega um modelo PPO e roda uma avaliação visual única."""
    print(f"--- 👁️  Iniciando Simulação Visual 3D ---")
    print(f"Carregando modelo de: {model_path}.zip")

    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo. {e}")
        return

    item_list = []
    if items:
        for item_str in items:
            try:
                dims = tuple(map(int, item_str.split('x')))
                item_list.append(dims)
            except:
                print(f"AVISO: Item '{item_str}' em formato inválido. Ignorando.")

    if not item_list:
        print("ERRO: Nenhum item válido para avaliar.")
        return

    eval_env = PalletizingEnv(item_list=item_list)
    obs, _ = eval_env.reset()

    plt.ion()
    done = False
    try:
        while not done:
            eval_env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nSimulação interrompida.")

    print("\n--- ✅ Simulação Visual Concluída ---")
    eval_env.render()
    plt.show(block=True)


def run_batch_validation(model_path, num_episodes, visualize_every=10):
    """
    Roda N episódios com listas de itens aleatórias e calcula as métricas médias.
    """
    # CORRIGIDO: A linha de impressão estava quebrada, agora está em uma única linha.
    print(f"--- ⚙️ Iniciando Validação em Lote com {num_episodes} Cenários Aleatórios ---")
    print(f"(Visualização a cada {visualize_every} testes)")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo. {e}")
        return

    all_boxes_placed = []
    all_occupancies = []

    eval_env = PalletizingEnv()

    plt.ion()

    for i in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        occupancy = np.sum(eval_env.pallet > 0) / np.prod(eval_env.pallet.shape)
        all_boxes_placed.append(eval_env.boxes_placed)
        all_occupancies.append(occupancy)

        if (i + 1) % visualize_every == 0:
            print(f"\n--- Visualização do Teste {i + 1}/{num_episodes} ---")
            eval_env.render()
            plt.pause(2) # Pausa um pouco mais longa para dar tempo de ver

        if (i + 1) % 10 == 0 or (i + 1) == num_episodes:
             print(f"Teste {i + 1}/{num_episodes} concluído...")

    plt.ioff()
    # Não vamos mais manter a última janela aberta para não travar o script
    # plt.show(block=True)

    print("\n--- 📊 Resultados Finais da Validação ---")
    print(f"Testes realizados: {num_episodes}")
    print(f"Caixas Colocadas (Média): {np.mean(all_boxes_placed):.2f} ± {np.std(all_boxes_placed):.2f}")
    print(f"Ocupação do Palete (Média): {np.mean(all_occupancies):.2%} ± {np.std(all_occupancies):.2%}")
    success_rate = (sum(1 for b in all_boxes_placed if b == eval_env.max_boxes) / num_episodes)
    print(f"Taxa de Sucesso (Paletes Completos): {success_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Painel de Controle para o Agente de Paletização.")
    parser.add_argument("--mode", type=str, choices=['train', 'eval', 'test'], required=True, help="Escolha o modo: 'train', 'eval' para visualização única, 'test' para teste em lote.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="[Modo Treino] Número de passos para treinar.")
    parser.add_argument("--save_path", type=str, default="ppo_palletizer_final", help="[Modo Treino] Caminho para salvar o modelo.")
    parser.add_argument("--load_path", type=str, default="ppo_palletizer_final", help="[Modo Avaliação/Teste] Caminho do modelo a ser carregado.")
    parser.add_argument('--items', nargs='+', help="[Modo Avaliação] Lista de volumes para a simulação visual única.")
    parser.add_argument("--runs", type=int, default=50, help="[Modo Teste] Número de testes aleatórios a serem executados.")
    parser.add_argument("--visualize_every", type=int, default=10, help="[Modo Teste] A cada quantos testes mostrar a visualização.")

    args = parser.parse_args()

    if args.mode == 'train':
        train_new_agent(timesteps=args.timesteps, save_path=args.save_path)

    elif args.mode == 'eval':
        if not args.items:
            print("ERRO: Para o modo 'eval', você deve fornecer os itens com o argumento --items.")
        else:
            evaluate_model_visual(model_path=args.load_path, items=args.items)

    elif args.mode == 'test':
        run_batch_validation(model_path=args.load_path, num_episodes=args.runs, visualize_every=args.visualize_every)