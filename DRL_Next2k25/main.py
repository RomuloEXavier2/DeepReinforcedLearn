import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from environment import PalletizingEnvironment
from agent import ImprovedDQNAgent

# O lookahead de 10, como você definiu, dará ao agente uma visão de futuro muito poderosa.
LOOKAHEAD_K = 10


def train_agent(episodes=20000, save_model=True, model_path="palletizing_agent.pth"):
    """
    Treina o agente de paletização com a configuração de ambiente definida.
    """
    # Instancia o ambiente com o lookahead
    env = PalletizingEnvironment(lookahead_k=LOOKAHEAD_K)

    state_size = len(env.get_state())
    action_size = env.pallet_size[0] * env.pallet_size[1] * env.max_height
    agent = ImprovedDQNAgent(state_size, action_size)

    scores = deque(maxlen=100)
    best_avg_score = float('-inf')

    print(f"Iniciando treinamento por {episodes} episódios...")
    print(f"Tamanho do Estado (com lookahead de {LOOKAHEAD_K}): {state_size} dimensões")
    print(f"Ações: {action_size} possíveis")
    print(f"Dispositivo: {agent.device}")
    print("=" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        # O loop 'while True' é mais robusto para episódios de duração variável
        while True:
            # O agente agora passa o 'env' para o 'act' para ter acesso a informações
            # como a lista de caixas para a Beam Search.
            action = agent.act(state, env)

            if action is None:
                # Isso acontece se get_valid_actions retornar uma lista vazia
                # print(f"Episódio {episode + 1}: Agente não encontrou ações válidas. Fim.")
                break

            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        agent.episode_rewards.append(total_reward)

        # A lógica de replay e salvamento continua a mesma
        if len(agent.memory) > agent.batch_size:
            agent.replay()

        avg_score = np.mean(scores)
        if len(scores) == 100 and avg_score > best_avg_score:
            best_avg_score = avg_score
            if save_model:
                agent.save_model(f"best_{model_path}")

        if (episode + 1) % 10 == 0:
            print(
                f"Episódio {episode + 1}/{episodes} | Recompensa Média (100 ep): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    if save_model:
        agent.save_model(model_path)

    return agent, env


def evaluate_agent(agent, env, episodes=5):
    """
    Avalia o agente treinado de forma simplificada, sem visualização 3D.
    """
    print(f"\n--- AVALIANDO AGENTE ({episodes} episódios) ---")
    agent.epsilon = 0.0  # Desliga a exploração
    all_scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Usa o modo padrão (não beam search) para uma avaliação rápida
            action = agent.act(state, env, use_beam_search=False)
            if action is None: break
            state, reward, done, info = env.step(action)
            total_reward += reward

        all_scores.append(total_reward)
        print(f"Avaliação {episode + 1}: Recompensa Final = {total_reward:.2f}, Caixas = {env.boxes_placed}")

    print(f"\nRecompensa Média na Avaliação: {np.mean(all_scores):.2f}")


def plot_training_results(agent):
    """Plota os resultados do treinamento."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resultados do Treinamento do Agente de Paletização', fontsize=16)

    # Recompensas
    axes[0, 0].plot(agent.episode_rewards)
    axes[0, 0].set_title('Recompensas por Episódio')
    axes[0, 0].grid(True)

    # Média Móvel
    if len(agent.episode_rewards) >= 100:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(100) / 100, mode='valid')
        axes[0, 1].plot(moving_avg, color='orange')
        axes[0, 1].set_title('Média Móvel (100 episódios)')
        axes[0, 1].grid(True)

    # Perda
    if agent.losses:
        axes[1, 0].plot(agent.losses, color='red')
        axes[1, 0].set_title('Perda (Loss)')
        axes[1, 0].grid(True)

    # Q-values
    if agent.q_values:
        axes[1, 1].plot(agent.q_values, color='green')
        axes[1, 1].set_title('Q-values Médios')
        axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def load_and_evaluate(model_path="best_palletizing_agent.pth"):
    """Carrega um modelo salvo e o avalia."""
    print(f"Tentando carregar modelo de: {model_path}")

    # Garante que o ambiente seja criado com os mesmos parâmetros do treino
    env = PalletizingEnvironment(lookahead_k=LOOKAHEAD_K)
    state_size = len(env.get_state())
    action_size = env.pallet_size[0] * env.pallet_size[1] * env.max_height
    agent = ImprovedDQNAgent(state_size, action_size)

    try:
        agent.load_model(model_path)
        print(f"Modelo carregado com sucesso.")
        evaluate_agent(agent, env, episodes=5)
    except FileNotFoundError:
        print(f"ERRO: Arquivo do modelo '{model_path}' não encontrado!")
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o modelo: {e}")


if __name__ == "__main__":
    while True:
        print("\n=== 📦 PAINEL DE CONTROLE - DeepPack3D (DQN Custom) ===")
        print("1. Treinar novo agente")
        print("2. Carregar e avaliar melhor agente salvo")
        print("3. Sair")

        choice = input("Escolha uma opção (1-3): ").strip()

        if choice == "1":
            trained_agent, trained_env = train_agent(episodes=20000, save_model=True)
            if trained_agent:
                print("\n--- Exibindo gráficos do treinamento ---")
                plot_training_results(trained_agent)
                print("\n--- Iniciando avaliação rápida do agente treinado ---")
                evaluate_agent(trained_agent, trained_env, episodes=5)

        elif choice == "2":
            model_file = input(f"Caminho do modelo (Enter para 'best_palletizing_agent.pth'): ").strip()
            if not model_file:
                model_file = "best_palletizing_agent.pth"
            load_and_evaluate(model_file)

        elif choice == "3":
            print("Encerrando o programa.")
            break

        else:
            print("Opção inválida! Por favor, escolha 1, 2 ou 3.")