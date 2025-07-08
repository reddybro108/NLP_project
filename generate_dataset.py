import csv
import random

positive_samples = [
    "I love using Copilot during my coding sessions!",
    "Copilot saves me so much time.",
    "Amazing tool, boosts my productivity.",
    "Feels like having a coding companion."
]

negative_samples = [
    "Copilot gave me totally wrong code.",
    "I hate how inaccurate suggestions are.",
    "Worst AI assistant ever.",
    "So frustrating when Copilot crashes."
]

neutral_samples = [
    "Copilot suggestion was okay.",
    "Not sure what to think about Copilot.",
    "Copilot is average at best.",
    "I have mixed feelings about Copilot."
]

labels = ['positive', 'negative', 'neutral']

with open('tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for _ in range(1000):
        label = random.choice(labels)
        if label == 'positive':
            text = random.choice(positive_samples)
        elif label == 'negative':
            text = random.choice(negative_samples)
        else:
            text = random.choice(neutral_samples)
        writer.writerow([text, label])
