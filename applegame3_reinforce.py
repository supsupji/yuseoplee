import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time

class AppleGame:
    def __init__(self, width=17, height=10, time_limit=120):
        self.width = width
        self.height = height
        self.time_limit = time_limit
        self.reset()
    
    def reset(self):
        # Initialize grid with random numbers from 1-9
        self.grid = np.random.randint(1, 10, size=(self.height, self.width))
        self.score = 0
        self.time_remaining = self.time_limit
        self.game_over = False
        return self.grid.copy()
    
    def is_valid_selection(self, x1, y1, x2, y2):
        # Check if coordinates are valid
        if not (0 <= x1 <= x2 < self.width and 0 <= y1 <= y2 < self.height):
            return False
        
        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        
        # Check if sum equals 10
        return np.sum(rectangle) == 10
    
    def get_rectangle_sum(self, x1, y1, x2, y2):
        # Ensure coordinates are valid
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        # Ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        
        # Return sum
        return np.sum(rectangle)
    
    def make_selection(self, x1, y1, x2, y2):
        if self.game_over:
            return 0, True
        
        # Ensure coordinates are valid
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        x2 = max(0, min(x2, self.width - 1))
        y2 = max(0, min(y2, self.height - 1))
        
        # Ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if not self.is_valid_selection(x1, y1, x2, y2):
            return 0, False
        
        # Extract the rectangle
        rectangle = self.grid[y1:y2+1, x1:x2+1]
        
        # Count apples in the rectangle
        num_apples = (rectangle > 0).sum()
        
        # Clear the apples (set to 0)
        self.grid[y1:y2+1, x1:x2+1] = 0
        
        # Update score
        self.score += num_apples
        
        # Check if all apples are cleared
        if (self.grid == 0).all():
            self.game_over = True
        
        return num_apples, True
    
    def update_time(self, dt):
        if self.game_over:
            return
        
        self.time_remaining -= dt
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True


class PyGameVisualizer:
    def __init__(self, game, cell_size=50):
        self.game = game
        self.cell_size = cell_size
        self.width = game.width * cell_size
        self.height = game.height * cell_size
        self.initialized = False
        self.selection_start = None
        self.selection_end = None
        
    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Apple Game")
        self.font = pygame.font.Font(None, 36)
        self.initialized = True
        
    def render(self):
        if not self.initialized:
            self.initialize()
            
        pygame.event.pump()
        pygame.display.flip()
        
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        for y in range(self.game.height):
            for x in range(self.game.width):
                value = self.game.grid[y, x]
                if value > 0:
                    # Draw apple
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    )
                    # Draw number
                    text = self.font.render(str(value), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(
                        x * self.cell_size + self.cell_size // 2,
                        y * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
                else:
                    # Draw empty cell
                    pygame.draw.rect(
                        self.screen,
                        (200, 200, 200),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                        1
                    )
        
        # Draw current selection
        if self.selection_start and self.selection_end:
            x1 = min(self.selection_start[0], self.selection_end[0])
            y1 = min(self.selection_start[1], self.selection_end[1])
            x2 = max(self.selection_start[0], self.selection_end[0])
            y2 = max(self.selection_start[1], self.selection_end[1])
            
            # Convert to grid coordinates
            grid_x1 = x1 // self.cell_size
            grid_y1 = y1 // self.cell_size
            grid_x2 = x2 // self.cell_size
            grid_y2 = y2 // self.cell_size
            
            # Calculate sum
            rect_sum = self.game.get_rectangle_sum(grid_x1, grid_y1, grid_x2, grid_y2)
            
            # Draw rectangle
            color = (0, 255, 0, 128) if rect_sum == 10 else (255, 0, 0, 128)
            
            # Convert back to pixel coordinates for drawing
            pixel_x1 = grid_x1 * self.cell_size
            pixel_y1 = grid_y1 * self.cell_size
            pixel_x2 = (grid_x2 + 1) * self.cell_size
            pixel_y2 = (grid_y2 + 1) * self.cell_size
            
            # Create a surface with alpha channel
            s = pygame.Surface((pixel_x2 - pixel_x1, pixel_y2 - pixel_y1), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (pixel_x1, pixel_y1))
            
            # Draw sum
            sum_text = self.font.render(f"Sum: {rect_sum}", True, (0, 0, 0))
            self.screen.blit(sum_text, (10, self.height - 40))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.game.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        
        # Draw time remaining
        time_text = self.font.render(f"Time: {int(self.game.time_remaining)}", True, (0, 0, 0))
        self.screen.blit(time_text, (self.width - 150, 10))
        
        pygame.display.flip()
    
    def set_selection(self, start_pos, end_pos):
        self.selection_start = start_pos
        self.selection_end = end_pos
    
    def clear_selection(self):
        self.selection_start = None
        self.selection_end = None
    
    def close(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False


class AppleGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, width=17, height=10, render_mode=None):
        super().__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Create the game
        self.game = AppleGame(width=width, height=height)
        
        # Define action space (top-left x, top-left y, rectangle width, rectangle height)
        self.action_space = spaces.MultiDiscrete([width, height, width, height])
        
        # Define observation space (grid of apples)
        self.observation_space = spaces.Box(low=0, high=9, shape=(height, width), dtype=np.int32)
        
        # Create visualizer if needed
        self.visualizer = None
        if render_mode == 'human':
            self.visualizer = PyGameVisualizer(self.game)
    
    def reset(self, seed=None, options=None):
        # Reset the game
        observation = self.game.reset()
        
        # Reset the renderer
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = PyGameVisualizer(self.game)
        
        return observation, {}
    
    def step(self, action):
        # Parse action
        x1, y1, rect_width, rect_height = action
        
        # Ensure width and height are at least 1
        rect_width = max(1, rect_width)
        rect_height = max(1, rect_height)
        
        # Calculate bottom-right coordinates
        x2 = min(x1 + rect_width - 1, self.width - 1)
        y2 = min(y1 + rect_height - 1, self.height - 1)
        
        # Make selection
        reward, valid = self.game.make_selection(x1, y1, x2, y2)
        
        # Update time (assume 1 second per step)
        if valid:
            self.game.update_time(1)
        
        # Adjust reward for RL
        if not valid:
            reward = -0.1  # Penalty for invalid selection
        
        # Small penalty for each step to encourage faster solving
        reward -= 0.01
        
        # Get observation
        observation = self.game.grid.copy()
        
        # Check if game is over
        done = self.game.game_over
        
        # Additional info
        info = {
            'score': self.game.score,
            'time_remaining': self.game.time_remaining,
            'valid_selection': valid
        }
        
        return observation, reward, done, False, info
    
    def render(self):
        if self.visualizer:
            self.visualizer.render()
    
    def close(self):
        if self.visualizer:
            self.visualizer.close()

def play_game(width=17, height=10):
    # Create the game
    game = AppleGame(width=width, height=height)
    visualizer = PyGameVisualizer(game)
    visualizer.initialize()
    
    # Variables for rectangle selection
    selecting = False
    
    clock = pygame.time.Clock()
    running = True
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start selection
                selecting = True
                visualizer.set_selection(event.pos, event.pos)
            elif event.type == pygame.MOUSEMOTION and selecting:
                # Update selection
                visualizer.set_selection(visualizer.selection_start, event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and selecting:
                # End selection
                selecting = False
                
                # Convert to grid coordinates
                x1 = visualizer.selection_start[0] // visualizer.cell_size
                y1 = visualizer.selection_start[1] // visualizer.cell_size
                x2 = event.pos[0] // visualizer.cell_size
                y2 = event.pos[1] // visualizer.cell_size
                
                # Ensure x1 <= x2 and y1 <= y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Make selection
                game.make_selection(x1, y1, x2, y2)
                
                # Clear selection
                visualizer.clear_selection()
        
        # Update time (assume 0.1 seconds per frame)
        game.update_time(0.1)
        
        # Render
        visualizer.render()
        
        # Cap at 10 FPS
        clock.tick(10)
    
    # Game over
    if game.game_over:
        font = pygame.font.Font(None, 72)
        text = font.render("Game Over!", True, (255, 0, 0))
        text_rect = text.get_rect(center=(visualizer.width // 2, visualizer.height // 2))
        visualizer.screen.blit(text, text_rect)
        
        # Show final score
        score_font = pygame.font.Font(None, 48)
        score_text = score_font.render(f"Final Score: {game.score}", True, (0, 0, 0))
        score_rect = score_text.get_rect(center=(visualizer.width // 2, visualizer.height // 2 + 50))
        visualizer.screen.blit(score_text, score_rect)
        
        pygame.display.flip()
        
        # Wait for user to close the window
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
    
    visualizer.close()









import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time

class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(PolicyNetwork, self).__init__()
        
        # 입력 차원 계산
        self.input_dim = obs_shape[0] * obs_shape[1]
        
        # 공통 특성 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 각 액션 차원에 대한 출력 헤드
        self.action_heads = nn.ModuleList()
        for dim_size in action_space.nvec:
            self.action_heads.append(nn.Linear(128, dim_size))
    
    def forward(self, x):
        # 입력을 1D로 평탄화하고 float 타입으로 변환
        x = x.float().view(-1, self.input_dim)
        
        # 특성 추출
        features = self.feature_extractor(x)
        
        # 각 액션 차원에 대한 로짓 계산
        action_logits = [head(features) for head in self.action_heads]
        
        return action_logits
    
    def get_action(self, state):
        # 상태를 PyTorch 텐서로 변환
        state_tensor = torch.tensor(state).float().view(1, -1)
        
        # 각 액션 차원에 대한 로짓 계산
        action_logits = self.forward(state_tensor)
        
        # 각 차원에 대한 확률 분포 생성
        action_dists = [Categorical(logits=logits) for logits in action_logits]
        
        # 각 차원에서 액션 샘플링
        actions = [dist.sample().item() for dist in action_dists]
        
        # 각 차원의 로그 확률 계산
        log_probs = [dist.log_prob(torch.tensor(act)) for dist, act in zip(action_dists, actions)]
        
        # 모든 로그 확률의 합을 반환
        return actions, torch.stack(log_probs).sum()
    
def collect_trajectories(env, policy, num_trajectories, steps_per_traj):
    trajectories = []
    
    for traj_idx in range(num_trajectories):
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': []
        }
        
        state, _ = env.reset()
        done = False
        t = 0
        
        while not done and t < steps_per_traj:
            # 상태 저장
            trajectory['states'].append(state)
            
            # 액션 선택
            action, log_prob = policy.get_action(state)
            
            # 액션 실행
            next_state, reward, done, _, info = env.step(action)
            
            # 액션, 보상, 로그 확률 저장
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['log_probs'].append(log_prob)
            
            # 상태 업데이트
            state = next_state
            t += 1
            
            # 렌더링
            env.render()
            time.sleep(0.05)
            
        print(f"Trajectory {traj_idx+1}/{num_trajectories}, Steps: {t}, Total Reward: {sum(trajectory['rewards']):.2f}")
        trajectories.append(trajectory)
    
    return trajectories
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    
    # 역순으로 returns 계산
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    return returns

def update_policy(policy, optimizer, trajectories, gamma=0.99):
    policy_loss = []
    
    for trajectory in trajectories:
        # returns 계산
        returns = compute_returns(trajectory['rewards'], gamma)
        returns = torch.FloatTensor(returns)
        
        #baseline
        returns = returns - returns.mean()
        
        # 표준화 (variance 감소)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 로그 확률 × 리턴의 합 계산
        for log_prob, R in zip(trajectory['log_probs'], returns):
            policy_loss.append(-log_prob * R)
    
    if policy_loss:
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        return policy_loss.item()
    return 0.0




def print_policy_parameters(policy):
    print("\nFinal Policy Parameters:")
    for name, param in policy.named_parameters():
        print(f"Layer: {name} | Size: {param.size()}")
        print(param.data)



def visualize_policy(policy, state):
    state_tensor = torch.tensor(state).float().view(1, -1)
    with torch.no_grad():
        action_logits = policy(state_tensor)
    action_probs = [torch.softmax(logits, dim=-1) for logits in action_logits]
    
    print("\nAction Probability Distribution:")
    for dim, probs in enumerate(action_probs):
        print(f"Dimension {dim}:")
        print(probs.numpy())


from torchsummary import summary

def print_model_architecture(policy, input_shape):
    print("\nModel Architecture:")
    summary(policy, input_shape)



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Apple Game')
    parser.add_argument('--width', type=int, default=17, help='Width of the grid')
    parser.add_argument('--height', type=int, default=10, help='Height of the grid')
    parser.add_argument('--mode', choices=['play', 'rl'], default='rl', help='Play mode or RL mode')
    
    args = parser.parse_args()
    
    if args.mode == 'play':
        # Play the game manually
        play_game(width=args.width, height=args.height)
    else:
        # RL mode demonstration
        env = AppleGameEnv(width=args.width, height=args.height, render_mode='human')
        print("Created RL environment. Use gym interface to interact with it.")
        
        
        
        # 하이퍼파라미터 설정
        N_updates = 20         # 총 업데이트 횟수
        N_trajectories = 5     # 각 업데이트마다 수집할 trajectory 개수
        steps_per_traj = 1000   # 각 trajectory의 최대 스텝 수

        # Policy network 및 optimizer 초기화
        policy = PolicyNetwork(env.observation_space.shape, env.action_space)
        optimizer = optim.Adam(policy.parameters(), lr=0.001)

        # 훈련 루프
        for update in range(N_updates):
            print(f"\n--- Update {update+1}/{N_updates} ---")
            
            # 1. N개의 trajectory 수집
            trajectories = collect_trajectories(env, policy, N_trajectories, steps_per_traj)
            
            # 2. 평균 보상 계산
            total_reward = sum([sum(traj['rewards']) for traj in trajectories])
            avg_reward = total_reward / N_trajectories
            
            # 3. 정책 업데이트
            loss = update_policy(policy, optimizer, trajectories, gamma=0.99)
            
             # 중간 진단 출력
            if (update+1) % 5 == 0:
                print(f"Update {update+1} Policy Sample:")
                sample_state = env.reset()[0]
                visualize_policy(policy, sample_state)
            
            print(f"Update {update+1}/{N_updates}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        print("\n=== Training Completed ===")
        print_policy_parameters(policy)
        print_model_architecture(policy, (env.observation_space.shape[0]*env.observation_space.shape[1],))

        # 학습 종료 후 환경 닫기
        env.close()

if __name__ == "__main__":
    main()