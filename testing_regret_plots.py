import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neural_exploration.contrastive_neural_ucb import ContrastiveNeuralUCB
from neural_exploration import *

def contrastive_UCB_test():
    T = int(5e2)
    n_arms = 4
    n_features = 8
    noise_std = 0.1

    confidence_scaling_factor = noise_std

    n_sim = 10

    SEED = 42
    p = 0.2
    hidden_size = 32
    epochs = 100
    train_every = 10
    use_cuda = False
    # Mean reward function
    a = np.random.randn(n_features)
    a /= np.linalg.norm(a, ord=2)
    # Linear Rewards
    reward_func_linear = lambda x: 10 * np.dot(a, x)
    reward_func_linear_pos = lambda x: 10 * np.dot(a, x) + .1 * np.random.randn()
    reward_func_linear_neg = lambda x: 10 * np.dot(a, x) + 5 * np.random.randn()

    reward_func_quad = lambda x: np.dot(a, x) + 100 * np.dot(a, x) ** 2
    reward_func_quad_pos = lambda x: np.dot(a, x) + 100 * np.dot(a, x) ** 2 + .1 * np.random.randn()
    reward_func_quad_neg = lambda x: np.dot(a, x) + 100 * np.dot(a, x) ** 2 + 5 * np.random.randn()

    # Cosine Reward
    reward_func_cos = lambda x: np.cos(2 * np.pi * np.dot(x, a))
    reward_func_cos_pos = lambda x: np.cos(2 * np.pi * np.dot(x, a)) + .1 * np.random.randn()
    reward_func_cos_neg = lambda x: np.cos(2 * np.pi * np.dot(x, a)) + 5 * np.random.randn()

    # Linear Contextual Bandits

    bandit_lin = ContextualBandit(T, n_arms, n_features, reward_func_linear, noise_std=noise_std, seed=SEED)

    bandit_lin_pos = ContextualBandit(T, n_arms, n_features, reward_func_linear_pos, noise_std=noise_std, seed=SEED)

    bandit_lin_neg = ContextualBandit(T, n_arms, n_features, reward_func_linear_neg, noise_std=noise_std, seed=SEED)

    regrets_lin = np.empty((n_sim, T))

    # Quadratic Contextual Bandits

    bandit_quad = ContextualBandit(T, n_arms, n_features, reward_func_quad, noise_std=noise_std, seed=SEED)

    bandit_quad_pos = ContextualBandit(T, n_arms, n_features, reward_func_quad_pos, noise_std=noise_std, seed=SEED)

    bandit_quad_neg = ContextualBandit(T, n_arms, n_features, reward_func_quad_neg, noise_std=noise_std, seed=SEED)

    regrets_quad = np.empty((n_sim, T))

    # Cosine Contextual Bandits
    bandit_cos = ContextualBandit(T, n_arms, n_features, reward_func_cos, noise_std=noise_std, seed=SEED)

    bandit_cos_pos = ContextualBandit(T, n_arms, n_features, reward_func_cos_pos, noise_std=noise_std, seed=SEED)

    bandit_cos_neg = ContextualBandit(T, n_arms, n_features, reward_func_cos_neg, noise_std=noise_std, seed=SEED)

    regrets_cos = np.empty((n_sim, T))


    # Linear Training
    for i in range(n_sim):
        bandit_lin.reset_rewards()
        bandit_lin_pos.reset_rewards()
        bandit_lin_neg.reset_rewards()
        model = ContrastiveNeuralUCB(bandit_lin,
                                    bandit_lin_pos,
                                    bandit_lin_neg,
                                    hidden_size=hidden_size,
                                     reg_factor=1.0,
                                     delta=0.1,
                                     confidence_scaling_factor=confidence_scaling_factor,
                                     training_window=100,
                                     p=p,
                                     learning_rate=0.01,
                                     epochs=epochs,
                                     train_every=train_every,
                                     use_cuda=use_cuda
                                 )
        model.run()
        regrets_lin[i] = np.cumsum(model.regrets)
    # Quad Training
    for i in range(n_sim):
        bandit_quad.reset_rewards()
        bandit_quad_pos.reset_rewards()
        bandit_quad_neg.reset_rewards()
        model = ContrastiveNeuralUCB(bandit_quad,
                                     bandit_quad_pos,
                                     bandit_quad_neg,
                                     hidden_size=hidden_size,
                                     reg_factor=1.0,
                                     delta=0.1,
                                     confidence_scaling_factor=confidence_scaling_factor,
                                     training_window=100,
                                     p=p,
                                     learning_rate=0.01,
                                     epochs=epochs,
                                     train_every=train_every,
                                     use_cuda=use_cuda
                                     )
        model.run()
        regrets_quad[i] = np.cumsum(model.regrets)
    # Cosine Training
    for i in range(n_sim):
        bandit_cos.reset_rewards()
        bandit_cos_pos.reset_rewards()
        bandit_cos_neg.reset_rewards()
        model = ContrastiveNeuralUCB(bandit_cos,
                                     bandit_cos_pos,
                                     bandit_cos_neg,
                                     hidden_size=hidden_size,
                                     reg_factor=1.0,
                                     delta=0.1,
                                     confidence_scaling_factor=confidence_scaling_factor,
                                     training_window=100,
                                     p=p,
                                     learning_rate=0.01,
                                     epochs=epochs,
                                     train_every=train_every,
                                     use_cuda=use_cuda
                                     )
        model.run()
        regrets_cos[i] = np.cumsum(model.regrets)
    return regrets_lin,regrets_quad,regrets_cos

def Neural_UCB_test():
    T = int(5e2)
    n_arms = 4
    n_features = 8
    noise_std = 0.1

    confidence_scaling_factor = noise_std

    n_sim = 10

    SEED = 42
    np.random.seed(SEED)
    p = 0.2
    hidden_size = 32
    epochs = 100
    train_every = 10
    use_cuda = False
    # Mean reward function
    a = np.random.randn(n_features)
    a /= np.linalg.norm(a, ord=2)
    reward_func_linear = lambda x: 10 * np.dot(a, x)
    reward_func_quad = lambda x: np.dot(a, x) + 100 * np.dot(a, x) ** 2
    reward_func_cos = lambda x: np.cos(2 * np.pi * np.dot(x, a))

    bandit_cos = ContextualBandit(T, n_arms, n_features, reward_func_cos, noise_std=noise_std, seed=SEED)
    bandit_quad = ContextualBandit(T, n_arms, n_features, reward_func_quad, noise_std=noise_std, seed=SEED)
    bandit_lin = ContextualBandit(T, n_arms, n_features, reward_func_linear, noise_std=noise_std, seed=SEED)
    regrets_cos = np.empty((n_sim, T))
    regrets_lin = np.empty((n_sim, T))
    regrets_quad = np.empty((n_sim, T))
    #Linear
    for i in range(n_sim):
        bandit_lin.reset_rewards()
        model = NeuralUCB(bandit_lin,
                          hidden_size=hidden_size,
                          reg_factor=1.0,
                          delta=0.1,
                          confidence_scaling_factor=confidence_scaling_factor,
                          training_window=100,
                          p=p,
                          learning_rate=0.01,
                          epochs=epochs,
                          train_every=train_every,
                          use_cuda=use_cuda
                                     )
        model.run()
        regrets_lin[i] = np.cumsum(model.regrets)
    #Quad
    for i in range(n_sim):
        bandit_quad.reset_rewards()
        model = NeuralUCB(bandit_quad,
                          hidden_size=hidden_size,
                          reg_factor=1.0,
                          delta=0.1,
                          confidence_scaling_factor=confidence_scaling_factor,
                          training_window=100,
                          p=p,
                          learning_rate=0.01,
                          epochs=epochs,
                          train_every=train_every,
                          use_cuda=use_cuda
                                     )
        model.run()
        regrets_quad[i] = np.cumsum(model.regrets)
    #Cosine
    for i in range(n_sim):
        bandit_cos.reset_rewards()
        model = NeuralUCB(bandit_cos,
                          hidden_size=hidden_size,
                          reg_factor=1.0,
                          delta=0.1,
                          confidence_scaling_factor=confidence_scaling_factor,
                          training_window=100,
                          p=p,
                          learning_rate=0.01,
                          epochs=epochs,
                          train_every=train_every,
                          use_cuda=use_cuda
                                     )
        model.run()
        regrets_cos[i] = np.cumsum(model.regrets)
    return regrets_lin, regrets_quad, regrets_cos


def LinUCB_Test():
    T = int(5e2)
    n_arms = 4
    n_features = 8
    noise_std = 0.1

    confidence_scaling_factor = noise_std

    n_sim = 10

    SEED = 42
    np.random.seed(SEED)

    # Mean reward function
    a = np.random.randn(n_features)
    a /= np.linalg.norm(a, ord=2)
    reward_func_linear = lambda x: 10 * np.dot(a, x)
    reward_func_quad = lambda x: np.dot(a, x) + 100 * np.dot(a, x) ** 2
    reward_func_cos = lambda x: np.cos(2 * np.pi * np.dot(x, a))

    bandit_cos = ContextualBandit(T, n_arms, n_features, reward_func_cos, noise_std=noise_std, seed=SEED)
    bandit_linear = ContextualBandit(T, n_arms, n_features, reward_func_linear, noise_std=noise_std, seed=SEED)
    bandit_quad = ContextualBandit(T, n_arms, n_features, reward_func_quad, noise_std=noise_std, seed=SEED)
    regrets_cos = np.empty((n_sim, T))
    regrets_quad = np.empty((n_sim, T))
    regrets_lin = np.empty((n_sim, T))
    # Linear
    for i in range(n_sim):
        bandit_linear.reset_rewards()
        model = LinUCB(bandit_linear)
        model.run()
        regrets_lin[i] = np.cumsum(model.regrets)
    # Quad
    for i in range(n_sim):
        bandit_quad.reset_rewards()
        model = LinUCB(bandit_quad)
        model.run()
        regrets_quad[i] = np.cumsum(model.regrets)
    # Cosine
    for i in range(n_sim):
        bandit_cos.reset_rewards()
        model = LinUCB(bandit_cos)
        model.run()
        regrets_cos[i] = np.cumsum(model.regrets)
    return regrets_lin, regrets_quad, regrets_cos


def plot_Contrastive_UCB(regrets_lin,regrets_neural,regrets_con,title,plot_title):
    fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)
    regrets_lin = np.load(regrets_lin)
    regrets_neural = np.load(regrets_neural)
    regrets_con = np.load(regrets_con)
    T = int(5e2)
    t = np.arange(T)

    mean_regrets_lin = np.mean(regrets_lin, axis=0)
    std_regrets_lin = np.std(regrets_lin, axis=0) / np.sqrt(regrets_lin.shape[0])

    mean_regrets_neural = np.mean(regrets_neural, axis=0)
    std_regrets_neural = np.std(regrets_neural, axis=0) / np.sqrt(regrets_neural.shape[0])

    mean_regrets_con = np.mean(regrets_con, axis=0)
    std_regrets_con = np.std(regrets_con, axis=0) / np.sqrt(regrets_con.shape[0])

    line1, = ax.plot(t, mean_regrets_lin)
    ax.fill_between(t, mean_regrets_lin - 2 * std_regrets_lin, mean_regrets_lin + 2 * std_regrets_lin, alpha=0.15)

    line2, = ax.plot(t, mean_regrets_neural)
    ax.fill_between(t, mean_regrets_neural - 2 * std_regrets_neural, mean_regrets_neural + 2 * std_regrets_neural, alpha=0.15)

    line3, = ax.plot(t, mean_regrets_con)
    ax.fill_between(t, mean_regrets_con - 2 * std_regrets_con, mean_regrets_con + 2 * std_regrets_con, alpha=0.15)
    ax.legend([line1, line2, line3], ['Linear_UCB', 'Neural_UCB', 'Contrast_UCB'])
    ax.set_title(plot_title)

    plt.tight_layout()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')


    plt.savefig(title, bbox_inches="tight")

def neural_ucb_debug():
    T = int(5e2)
    n_arms = 4
    n_features = 8
    noise_std = 0.1

    confidence_scaling_factor = noise_std

    n_sim = 3

    SEED = 42
    np.random.seed(SEED)

    p = 0.2
    hidden_size = 32
    epochs = 100
    train_every = 10
    use_cuda = False
    a = np.random.randn(n_features)
    a /= np.linalg.norm(a, ord=2)
    reward_func = lambda x: np.cos(2 * np.pi * np.dot(x, a))
    bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)

    regrets = np.empty((n_sim, T))

    for i in range(n_sim):
        bandit.reset_rewards()
        model = NeuralUCB(bandit,
                          hidden_size=hidden_size,
                          reg_factor=1.0,
                          delta=0.1,
                          confidence_scaling_factor=confidence_scaling_factor,
                          training_window=100,
                          p=p,
                          learning_rate=0.01,
                          epochs=epochs,
                          train_every=train_every,
                          use_cuda=use_cuda,
                          )
        model.run()
        regrets[i] = np.cumsum(model.regrets)
if __name__ == '__main__':

    # Contrastive Regrets
    title = 'figures/cosine_reward.png'
    plot_title = 'Regret with Cosine Reward Function'
    linear_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_linear_cos.npy'
    neural_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_neural_cos.npy'
    con_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_contrast_cos.npy'
    # Plotting
    plot_Contrastive_UCB(linear_path, neural_path, con_path, title,plot_title)

    # Contrastive Regrets
    title = 'figures/linear_reward.png'
    plot_title = 'Regret with Linear Reward Function'
    linear_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_linear_lin.npy'
    neural_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_neural_lin.npy'
    con_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_contrast_lin.npy'
    # Plotting
    plot_Contrastive_UCB(linear_path, neural_path, con_path, title,plot_title)

    # Contrastive Regrets
    title = 'figures/quad_reward.png'
    plot_title = 'Regret with Quadratic Reward Function'
    linear_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_linear_quad.npy'
    neural_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_neural_quad.npy'
    con_path = '/home/kiran/Desktop/Dev/neural_exploration/regrets_contrast_quad.npy'
    # Plotting
    plot_Contrastive_UCB(linear_path, neural_path, con_path, title,plot_title)
    '''
    # Experiment

    print('Linear')
    regrets_lin, regrets_quad, regrets_cos = LinUCB_Test()
    np.save('regrets_linear_lin.npy', regrets_lin)
    np.save('regrets_linear_quad.npy', regrets_quad)
    np.save('regrets_linear_cos.npy', regrets_cos)
    
    print('Contrastive')
    regrets_lin, regrets_quad, regrets_cos = contrastive_UCB_test()
    np.save('regrets_contrast_lin.npy', regrets_lin)
    np.save('regrets_contrast_quad.npy', regrets_quad)
    np.save('regrets_contrast_cos.npy', regrets_cos)
    

    print('Neural')
    # Neural Regrets
    regrets_lin, regrets_quad, regrets_cos = Neural_UCB_test()
    np.save('regrets_neural_lin.npy', regrets_lin)
    np.save('regrets_neural_quad.npy', regrets_quad)
    np.save('regrets_neural_cos.npy', regrets_cos)
    '''
