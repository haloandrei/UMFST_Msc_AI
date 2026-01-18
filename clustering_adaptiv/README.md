### Clustering adaptiv de imagini folosind CNN, K-means si Reinforcement Learning

Dataset: https://www.kaggle.com/datasets/moltean/fruits

   source .venv/bin/activate


Rulare
- Baseline:
  python -m src.baseline 

- RL adaptiv:
  python -m src.train_rl 
   --k-values 8,10,12,15

 Proiectul foloseste XPU (Intel)