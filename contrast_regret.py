import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neural_exploration.contrastive_neural_ucb import ContrastiveNeuralUCB
from neural_exploration import *
import pandas as pd

# initialize list of lists
plt.figure(1)
data = [['SupCon', 0], ['SimCLR', 25.136]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Method', 'Regret'])

sns.barplot(x = 'Method',
            y = 'Regret',
            data = df,
            ci = 0)

plt.title('Contrastive Regret of each Method Cifar-10')
plt.grid()
plt.savefig('idea.png')

plt.figure(2)
# initialize list of lists
data = [['SupCon', 0], ['SimCLR', 2.138]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Method', 'Regret'])

sns.barplot(x = 'Method',
            y = 'Regret',
            data = df,
            ci = 0)

plt.title('Contrastive Regret of each Method Cifar-100')
plt.grid()
plt.savefig('idea2.png')



