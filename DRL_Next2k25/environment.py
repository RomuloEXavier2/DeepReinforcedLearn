import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PalletizingEnvironment:
    def __init__(self, pallet_size=(5, 5), max_height=5, item_list=None, lookahead_k=3):
        self.pallet_size = pallet_size
        self.max_height = max_height
        self.initial_item_list = item_list
        self.lookahead_k = lookahead_k
        self.fig = None
        self.ax = None
        self.reset()

    def reset(self):
        self.pallet = np.zeros((*self.pallet_size, self.max_height), dtype=int)
        self.max_boxes = 25
        if self.initial_item_list:
            self.items_to_place = self.initial_item_list.copy()
            self.max_boxes = len(self.items_to_place)
        else:
            self.items_to_place = [self.generate_box() for _ in range(self.max_boxes)]
        self.current_box = self.get_next_box()
        self.boxes_placed = 0
        self.step_count = 0
        self.max_steps = self.max_boxes + 20
        self.placed_box_properties = []
        return self.get_state()

    def get_next_box(self):
        return self.items_to_place.pop(0) if self.items_to_place else None

    def generate_box(self):
        sizes = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (1, 3, 3),
                 (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 1), (2, 2, 2), (2, 2, 3),
                 (2, 3, 1), (2, 3, 2), (2, 3, 3), (3, 1, 1), (3, 1, 2), (3, 1, 3),
                 (3, 2, 1), (3, 2, 2), (3, 2, 3), (3, 3, 1), (3, 3, 2), (3, 3, 3)]
        return random.choice(sizes)

    def get_state(self):
        state = []
        state.extend(np.sum(self.pallet > 0, axis=2).flatten())
        state.extend(self.current_box if self.current_box else (0, 0, 0))
        for i in range(self.lookahead_k):
            state.extend(self.items_to_place[i] if i < len(self.items_to_place) else [0, 0, 0])
        total_occupied = np.sum(self.pallet > 0)
        state.append(total_occupied / np.prod(self.pallet.shape))
        state.append(self.boxes_placed / self.max_boxes if self.max_boxes > 0 else 0)
        state.append((self.max_steps - self.step_count) / self.max_steps)
        return np.array(state, dtype=np.float32)

    def is_valid_placement(self, x, y, z, box_size):
        if not box_size: return False
        bx, by, bz = box_size
        if x + bx > self.pallet_size[0] or y + by > self.pallet_size[1] or z + bz > self.max_height: return False
        if np.any(self.pallet[x:x + bx, y:y + by, z:z + bz] > 0): return False
        if z > 0 and np.sum(self.pallet[x:x + bx, y:y + by, z - 1] > 0) < (bx * by * 0.7): return False
        return True

    def place_box(self, x, y, z, box_size):
        bx, by, bz = box_size
        box_id = self.boxes_placed + 1
        self.pallet[x:x + bx, y:y + by, z:z + bz] = box_id
        self.placed_box_properties.append({'pos': (x, y, z), 'dims': box_size, 'id': box_id})
        self.boxes_placed += 1

    def step(self, action):
        self.step_count += 1
        if not self.current_box:
            return self.get_state(), 0, True, {'placed': False, 'reason': 'no_more_boxes'}

        total_positions = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        if action >= total_positions:
            return self.get_state(), -25, True, {'placed': False, 'reason': 'invalid_action_index'}

        x, y, z = action // (self.pallet_size[1] * self.max_height), (
                    action % (self.pallet_size[1] * self.max_height)) // self.max_height, action % self.max_height

        C, L, A = self.current_box
        orientations = set([(C, L, A), (L, C, A), (C, A, L), (A, C, L), (L, A, C), (A, L, C)])

        box_to_place = None
        for orientation in orientations:
            if self.is_valid_placement(x, y, z, orientation):
                box_to_place = orientation
                break

        if box_to_place is not None:
            self.place_box(x, y, z, box_to_place)
            reward = 50
            self.current_box = self.get_next_box()
            info = {'placed': True, 'reason': 'success'}
        else:
            reward = -5
            info = {'placed': False, 'reason': 'invalid_placement_all_orientations'}

        done = (not self.current_box or self.step_count >= self.max_steps)
        if done and self.boxes_placed == self.max_boxes: reward += 500

        return self.get_state(), reward, done, info

    def render(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 8));
            self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.clear()
        colors = plt.cm.get_cmap('viridis', self.max_boxes + 1)
        for box in self.placed_box_properties:
            x, y, z = box['pos'];
            dx, dy, dz = box['dims'];
            box_id = box['id']
            self.ax.bar3d(x, y, z, dx, dy, dz, color=colors(box_id), edgecolor='black', alpha=0.8)
        self.ax.set_xlabel('X');
        self.ax.set_ylabel('Y');
        self.ax.set_zlabel('Z (Altura)')
        self.ax.set_xlim([0, self.pallet_size[0]]);
        self.ax.set_ylim([0, self.pallet_size[1]]);
        self.ax.set_zlim([0, self.max_height])
        self.ax.set_title(f"Palete 3D | Caixas: {self.boxes_placed}/{self.max_boxes}")
        if self.current_box:
            dx, dy, dz = self.current_box
            self.ax.bar3d(-dx - 1, 0, 0, dx, dy, dz, color='red', edgecolor='black', alpha=0.5)
            self.ax.text(-dx - 1, 0, dz + 1, "Próxima Caixa", color='red')
        plt.draw();
        plt.pause(0.01)