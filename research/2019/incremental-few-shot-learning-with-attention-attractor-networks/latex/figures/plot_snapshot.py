from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
  fig, ax = plt.subplots(figsize=(7, 5))
  data = pd.read_csv('snapshot.csv')
  print(data)
  models = ['Chance', 'LR', 'LR +S', 'LR +A', 'MLP', 'MLP +S', 'MLP +A']
  colors = ['Gray', 'LightBlue', 'SkyBlue', 'SteelBlue', 'LightCoral', 'IndianRed', 'DarkRed']
  ind = np.arange(4) * 1.5
  width = 0.2      # the width of the bars
  gap = 0.05
  # scores = data[:, ::2]
  # print('all scores', scores)
  ax.grid((0.8, 0.8, 0.8), linestyle='--', linewidth=0.5)
  for ii, model in enumerate(models):
    score = data[model]
    pm = data['+/-' + ('.{}'.format(ii) if ii > 0 else '')]
    print(model)
    print('score', score)
    print('pm', pm)
    rects = ax.bar(ind + ii * width, score, width - gap, color=colors[ii], yerr=pm)

  ax.set_ylim(48,58)
  ax.set_ylabel('Accuracy (%)', fontsize=20)
  ax.set_xlabel('Number of Base Classes', fontsize=20)
  ax.set_xticks(ind + 0.5 * width * (len(models) - 1.0))
  ax.set_yticklabels(('48', '50', '52', '54', '56', '58'), fontsize=16)
  ax.set_xticklabels(('50', '100', '150', '200'), fontsize=16)
  ax.legend(models, ncol=2, fontsize=14)
  plt.tight_layout()
  plt.savefig('snapshot.pdf')
  plt.show()


if __name__ == '__main__':
  main()
