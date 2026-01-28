import numpy as np

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box

from scipy.stats import multivariate_normal


## create a wrapper that makes the noisy observations, so we have direct access to the pdf (replace TransformObservation) -- will need to vectorize the likelihood calculation

class ParticleFilter(Wrapper):
    def __init__(self, env, num_particles:int=10):
        super().__init__(env)
        self.num_particles = num_particles

        env_id = env.unwrapped.spec.id
        def make_env():
            def _thunk():
                base_env = gym.make(env_id)
                wrapped_env = POMDParticle(base_env)
                return wrapped_env
            return _thunk   

        self.particle_envs = gym.vector.SyncVectorEnv([make_env() for _ in range(self.num_particles)])

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation = obs

        self._particle_filter(action)

        info['estimated_state'] = self.state_estimate

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.observation, _ = self.env.reset()
        self.particle_states, _ = self.particle_envs.reset()
        self.weights = np.ones(self.num_particles) / self.num_particles

        return


    def _particle_filter(self, action):

        # step 1: advance particles using vectorized env step
        
        _, new_infos = self._particle_step(action) 

        new_states = np.array(new_infos['true_state'])  # ensure numpy array (num_particles, state_dim)

        # step 2: compute weights based on observation likelihood (vectorized)
        # try raising likelihood to the power of 1/self.env.observation_space.shape[0]
        new_weights_unnormalized = np.array([
            weight*self.env._obs_likelihood(self.observation, state)**(1/self.particle_envs.single_observation_space.shape[0]) # type: ignore
            for (state,weight) in zip(new_states, self.weights)
        ])
        new_weights_unnormalized = np.maximum(new_weights_unnormalized, 1e-10)  # avoid numerical issues
        new_weights = new_weights_unnormalized / new_weights_unnormalized.sum()

        # Vectorized state estimate computation
        self.state_estimate = (new_weights.reshape(-1,1) * new_states).sum(axis=0)

        # step 3: resampling
        # Compute effective sample size
        Neff = 1.0 / (new_weights ** 2).sum()
        # resample if this condition is met
        if Neff < (self.num_particles // 3):
            resampled_state_idx = np.random.choice(
                np.arange(self.num_particles), 
                self.num_particles, 
                p=new_weights
            )
            new_states = new_states[resampled_state_idx]
            new_weights = np.ones(self.num_particles) / self.num_particles

        self.particle_states = new_states

        
        # Update particle environment states
        self.particle_envs.set_attr("state", [state.copy() for state in new_states])
        
        self.weights = new_weights

        return 


    def _particle_step(self, action):
        # Step all particle environments with the same action
        # print(np.tile(action, (self.num_particles,  env.action_space.shape[0] )))
        if env.action_space.shape == ():
            tiled_action = np.tile(action, (self.num_particles,))
        else:
            tiled_action =  np.tile(action, (self.num_particles, 1 ))
        obs, _, _, _, info = self.particle_envs.step(tiled_action)
        return obs, info



## Observation wrapper to apply transformation to state and add noise
# This enables us to directly compute the observation likelihoods downstream
# Note an observation wrapper should be designed individually for specific environments; this structure
# decouples the environment (uncertain and base) from downstream wrappers 

class POMDParticle(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.pdf = multivariate_normal
        self.obs_cov = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # observation noise covariance

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)    

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        trans_state = self._transform_state(state)
        obs = self._observation(trans_state)

        info['true_state'] = state  # include true state in info for reference

        return obs, reward, False, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, info = self.env.reset()

        info['true_state'] = state  # include true state in info for reference

        return self._observation(self._transform_state(state)), info

    def _transform_state(self, state):
        # define some function of the state

        return state[[0,1,2,3,4,5,8]]

    def _observation(self, transformed_state):
        # sample from observation model

        return self.pdf.rvs(mean=transformed_state, cov=self.obs_cov)

    def _obs_likelihood(self, observation, state):
        # Compute the likelihood of the observation given a state (particle)
        mean = self._transform_state(state)
        obs_prob = self.pdf(mean=mean, cov=self.obs_cov).pdf(observation)

        return obs_prob
    

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    ## define true environment
    env = gym.make("InvertedDoublePendulum-v4")  # base environment
    obs_env = POMDParticle(env) # noisy, true environment--pass to particle wrapper

    # define environments to act as transition model for particle filter
    num_particles = 50
    # env_particles = gym.make_vec("Acrobot-v1", num_envs=num_particles, vectorization_mode="async") ## create inside particle wrapper based on base environment

    wrapped_env = ParticleFilter(obs_env, num_particles=num_particles)

    obs = wrapped_env.reset()
    
    fig, ax = plt.subplots(2, sharex=True, layout='constrained')
    state_history = []
    for _ in range(50):
        action = obs_env.action_space.sample()
        # actions = np.repeat(action, num_particles, axis=0)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        state_history.append([info['estimated_state'], info['true_state']])
        # print("true", info['true_state'])
        # print("estimate", info['estimated_state'])

    ax[0].plot([s[1][6] for s in state_history], label='True vel1')
    ax[0].plot([s[0][6] for s in state_history], label='Estimated vel1')
    ax[1].plot([s[1][7] for s in state_history], label='True vel2')
    ax[1].plot([s[0][7] for s in state_history], label='Estimated vel2')
    ax[0].legend()
    plt.show()