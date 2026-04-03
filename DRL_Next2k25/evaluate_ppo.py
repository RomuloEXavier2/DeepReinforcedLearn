import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import PalletizingEnv  # Importa nosso ambiente final


def parse_items(item_strings):
    # ... (função sem alterações) ...
    item_list = [];
    if item_strings:
        for item_str in item_strings:
            try:
                dims = tuple(map(int, item_str.split('x'))); item_list.append(dims)
            except:
                print(f"AVISO: Dimensão inválida '{item_str}'.")
    return item_list


def run_visual_evaluation(model_path, items, pallet_size, height, lookahead):
    # ... (esta função de visualização única não muda) ...
    print(f"--- Carregando modelo PPO de '{model_path}.zip' ---")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo. {e}"); return
    item_list = parse_items(items)
    if not item_list: print("ERRO: Nenhum item válido fornecido."); return
    eval_env = PalletizingEnv(pallet_size=pallet_size, max_height=height, item_list=item_list, lookahead_k=lookahead)
    obs, info = eval_env.reset()
    print("\n--- 👁️  Iniciando Simulação Visual 3D ---")
    plt.ion();
    done = False
    try:
        while not done:
            eval_env.render()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nSimulação interrompida.")
    print("\n--- ✅ Simulação Concluída ---");
    eval_env.render();
    plt.ioff();
    plt.show()


# --- NOVO: FUNÇÃO PARA TESTE DE ROBUSTEZ ---
def run_batch_test(model_path, num_runs, pallet_size, height, lookahead):
    """
    Executa múltiplos testes com listas de itens aleatórias para validar a generalização.
    """
    print(f"--- ⚙️ Iniciando Teste de Generalização com {num_runs} Cenários Aleatórios ---")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o modelo. {e}"); return

    all_boxes_placed = []
    all_occupancies = []

    for i in range(num_runs):
        print(f"Rodando teste {i + 1}/{num_runs}...")
        # Cria um ambiente com uma nova lista aleatória de 25 caixas
        env = PalletizingEnv(pallet_size=pallet_size, max_height=height, lookahead_k=lookahead)
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        occupancy = np.sum(env.pallet > 0) / np.prod(env.pallet.shape)
        all_boxes_placed.append(env.boxes_placed)
        all_occupancies.append(occupancy)

    print("\n--- 📊 Resultados Finais da Validação ---")
    print(f"Caixas Colocadas (Média): {np.mean(all_boxes_placed):.2f} ± {np.std(all_boxes_placed):.2f}")
    print(f"Ocupação do Palete (Média): {np.mean(all_occupancies):.2%} ± {np.std(all_occupancies):.2%}")
    success_rate = (sum(1 for b in all_boxes_placed if b == env.max_boxes) / num_runs)
    print(f"Taxa de Sucesso (Paletes Completos): {success_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia um agente de paletização treinado.")
    parser.add_argument("--model", type=str, default="ppo_palletizing_agent")
    parser.add_argument('--items', nargs='+', help="[MODO VISUAL] Lista de volumes. Formato: LxWxH")
    # NOVO: Argumento para o modo de teste em lote
    parser.add_argument("--random_test", type=int, metavar='N',
                        help="[MODO TESTE] Executa N testes com listas aleatórias.")

    # Parâmetros do ambiente (usado em ambos os modos)
    parser.add_argument("--width", type=int, default=5);
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--height", type=int, default=5);
    parser.add_argument("--lookahead", type=int, default=3)

    args = parser.parse_args()

    # Decide qual modo executar
    if args.random_test:
        run_batch_test(
            model_path=args.model,
            num_runs=args.random_test,
            pallet_size=(args.width, args.depth),
            height=args.height,
            lookahead=args.lookahead
        )
    elif args.items:
        run_visual_evaluation(
            model_path=args.model,
            items=args.items,
            pallet_size=(args.width, args.depth),
            height=args.height,
            lookahead=args.lookahead
        )
    else:
        print(
            "Erro: Você deve fornecer ou uma lista de itens com '--items' ou um número de testes com '--random_test'.")
        parser.print_help()