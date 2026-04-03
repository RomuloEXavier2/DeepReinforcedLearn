import torch
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import copy
from environment import PalletizingEnvironment
from agent import ImprovedDQNAgent


def parse_items(item_strings):
    item_list = [];
    if item_strings:
        for item_str in item_strings:
            try:
                dims = tuple(map(int, item_str.split('x'))); item_list.append(dims)
            except:
                print(f"AVISO: Dimensão inválida '{item_str}'.")
    return item_list


def run_visual_evaluation(model_path, items, height, lookahead, use_beam_search, beam_width):
    if use_beam_search:
        print("--- Simulação com Planejamento (Beam Search) ---")
    else:
        print("--- Simulação (Modo Padrão) ---")

    item_list = parse_items(items)
    if not item_list: print("ERRO: Nenhum volume válido fornecido."); return

    env = PalletizingEnvironment(max_height=height, item_list=item_list, lookahead_k=lookahead)
    state_size = len(env.get_state());
    action_size = env.pallet_size[0] * env.pallet_size[1] * env.max_height
    agent = ImprovedDQNAgent(state_size, action_size)
    try:
        agent.load_model(model_path); print(f"Modelo '{model_path}' carregado.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}"); return

    agent.epsilon = 0.0
    plt.ion()
    print("\nPressione Ctrl+C no terminal para parar a simulação.")
    try:
        state = env.reset();
        done = False
        while not done:
            env.render()
            action = agent.act(state, env, use_beam_search=use_beam_search, beam_width=beam_width)
            if action is None: print("Agente não encontrou ações válidas. Fim."); break
            state, reward, done, info = env.step(action)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nSimulação interrompida.")
    print("\n--- Simulação Concluída ---");
    env.render();
    plt.ioff();
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza um agente de paletização treinado em 3D.")
    parser.add_argument("--model", type=str, default="best_palletizing_agent.pth")
    parser.add_argument("--items", nargs='+', required=True, help="Lista de volumes. Formato: LxWxH")
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--beam", action='store_true', help="Ativa a Beam Search.")
    parser.add_argument("--beam_width", type=int, default=3, help="Largura da Beam Search (k).")
    args = parser.parse_args()
    run_visual_evaluation(model_path=args.model, items=args.items, height=args.height, lookahead=args.lookahead,
                          use_beam_search=args.beam, beam_width=args.beam_width)