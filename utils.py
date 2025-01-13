import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import deque

class QLearningRegressor:
    def __init__(self):
        # Initialize model with SGD for online learning
        self.model = SGDRegressor(learning_rate="constant", eta0=0.01)
        self.fitted = False
        
        # Q-learning parameters
        self.gamma = 0.99  # discount factor
        
        # Feature extraction parameters
        self.hour_window = 12
        self.day_window = 24 * 5
        self.week_window = 24 * 7 * 3
        
        # Price buffers
        self.hour_buffer = deque(maxlen=self.hour_window)
        self.day_buffer = deque(maxlen=self.day_window)
        self.week_buffer = deque(maxlen=self.week_window)
        
        # Initialize first fit flag
        self.is_first_fit = True
        
    def update_buffers(self, price):
        """Update all price buffers"""
        self.hour_buffer.append(price)
        self.day_buffer.append(price)
        self.week_buffer.append(price)
        
    def get_moving_averages(self):
        """Calculate moving averages for different timeframes"""
        hour_ma = np.mean(self.hour_buffer) if len(self.hour_buffer) > 0 else 0
        day_ma = np.mean(self.day_buffer) if len(self.day_buffer) > 0 else 0
        week_ma = np.mean(self.week_buffer) if len(self.week_buffer) > 0 else 0
        return hour_ma, day_ma, week_ma
    
    def extract_features(self, state, action):
        """Extract features for Q-function approximation"""
        storage_level, current_price, hour, day = state
        self.update_buffers(current_price)
        
        hour_ma, day_ma, week_ma = self.get_moving_averages()
        
        features = np.array([
            storage_level,
            current_price,
            hour,
            action,
            hour_ma,
            day_ma,
            week_ma,
            current_price - hour_ma,  # Price deviation from hourly MA
            current_price - day_ma,   # Price deviation from daily MA
            current_price - week_ma,  # Price deviation from weekly MA
            storage_level * current_price,  # Interaction term
            float(hour <= 12),  # Morning indicator
            float(current_price > hour_ma),  # Price above hourly MA indicator
            float(storage_level < 60)  # Low storage indicator
        ])
        
        return features.reshape(1, -1)
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        if not self.fitted:
            return 0.0
        features = self.extract_features(state, action)
        return self.model.predict(features)[0]
    
    def get_action(self, state, storage_level):
        """Get action using Q-value maximization"""
        # Consider discrete actions for computational efficiency
        actions = np.linspace(-1, 1, 21)  # 21 actions between -1 and 1
        
        q_values = [self.get_q_value(state, a) for a in actions]
        best_action = actions[np.argmax(q_values)]
        
        # Safety checks for storage requirements
        # _, _, hour, _ = state
        # hours_left = 24 - hour
        # if hours_left > 0:
        #     energy_needed = 120 - storage_level
        #     min_required_rate = energy_needed / hours_left
        #     if min_required_rate > 8:  # Leave some margin
        #         return 1.0
            
        return best_action
        
    def update(self, state, action, reward, next_state):
        """Update Q-function using Bellman equation"""
        # Get features for current state-action
        features = self.extract_features(state, action)
        
        # Calculate target using Bellman equation
        # Q(s,a) = R + γ * max_a' Q(s',a')
        if self.fitted:
            # Get maximum Q-value for next state
            actions = np.linspace(-1, 1, 21)
            next_q_values = [self.get_q_value(next_state, a) for a in actions]
            max_next_q = max(next_q_values)
            
            # Bellman equation
            target = reward + self.gamma * max_next_q
        else:
            target = reward
            
        # Fit model
        if self.is_first_fit:
            self.model.partial_fit(features, [target])
            self.is_first_fit = False
            self.fitted = True
        else:
            self.model.partial_fit(features, [target])