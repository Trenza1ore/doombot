import numpy as np
import matplotlib.pyplot as plt
from models import CombatAgent
from discord_webhook import discord_bot

def plot_stat(train_scores: list, all_scores:list, train_quartiles: list, 
              epoch: int, agent: CombatAgent, bot: discord_bot, epoch_start: int=0, plot: bool=True) -> str:
    train_scores = np.array(train_scores)
    Q1, Q2, Q3 = np.percentile(train_scores, 25), np.median(train_scores), np.percentile(train_scores, 75)
    mean = train_scores.mean()
    
    train_quartiles[0].append(Q1)
    train_quartiles[1].append(Q2)
    train_quartiles[2].append(Q3)
    train_quartiles[3].append(mean)
    
    if plot:
        plt.clf()
        
        plt.bar(np.arange(len(train_scores)), train_scores, 1)
        plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
        plt.xlim(0, len(train_scores)+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.savefig(f"{bot.path}/{epoch}.png")
        plt.clf()
        
        plt.bar(np.arange(len(all_scores[0])), all_scores[0], 1)
        plt.title("All epoches")
        plt.xlim(0, len(all_scores[0])+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        plt.savefig(f"{bot.path}/{epoch}a.png")
        plt.clf()
        
        quartile_x = np.arange(len(train_quartiles[0])) + epoch_start + 1
        plt.plot(quartile_x, train_quartiles[0], 'r--', label="Q1")
        plt.plot(quartile_x, train_quartiles[1], 'k-', label="Q2")
        plt.plot(quartile_x, train_quartiles[2], 'b--', label="Q3")
        plt.plot(quartile_x, train_quartiles[3], 'k--', label="Mean")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("Quartiles")
        plt.legend()
        plt.xlim(0, len(train_quartiles[0])+1)
        plt.savefig(f"{bot.path}/train_quartiles.png")
        plt.clf()
        
        kd_ratio_x = np.arange(len(all_scores[1]))
        kd_ratio = np.cumsum(all_scores[1])/(kd_ratio_x+1)
        plt.plot(kd_ratio_x, kd_ratio)
        plt.title("Average K/D Score")
        plt.xlabel("episode")
        plt.ylabel("kill/death ratio")
        plt.savefig(f"{bot.path}/train_kill_counts.png")
        plt.clf()
        
        plt.subplot(2, 2, 1)
        plt.bar(np.arange(len(train_scores)), train_scores, 1)
        plt.title(f"Epoch {epoch+1}, eps = {agent.eps}")
        plt.xlim(0, len(train_scores)+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        
        plt.subplot(2, 2, 2)
        plt.bar(np.arange(len(all_scores[0])), all_scores[0], 1)
        plt.title("All epoches")
        plt.xlim(0, len(all_scores[0])+1)
        plt.xlabel("episode")
        plt.ylabel("score")
        
        plt.subplot(2, 2, 3)
        quartile_x = np.arange(len(train_quartiles[0])) + epoch_start + 1
        plt.plot(quartile_x, train_quartiles[0], 'r--', label="Q1")
        plt.plot(quartile_x, train_quartiles[1], 'k-', label="Q2")
        plt.plot(quartile_x, train_quartiles[2], 'b--', label="Q3")
        plt.plot(quartile_x, train_quartiles[3], 'c-', label="Mean")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("Quartiles")
        plt.legend()
        plt.xlim(0, len(train_quartiles[0])+1)
        
        plt.subplot(2, 2, 4)
        plt.plot(kd_ratio_x, kd_ratio)
        plt.title("Average K/D Score")
        plt.xlabel("episode")
        plt.ylabel("kill/death ratio")
        plt.xlim(0, len(all_scores[1])+1)
        plt.ylim(-1, 7)
        
        plt.savefig(f"{bot.path}/current.png")
        plt.clf()
    
    np.save(f"{bot.path}/train_quartiles.npy", np.asfarray(train_quartiles))
    np.save(f"{bot.path}/train_kill_counts.npy", np.asfarray(all_scores[1]))
    np.save(f"{bot.path}/scores_{epoch}.npy", train_scores)
    np.save(f"{bot.path}/scores_all_{epoch}.npy", np.asfarray(all_scores[0]))
    
    stats = f"Result:\nmean: {mean:.2f} +- {train_scores.std():.2f}\nQ1: {Q1}\nQ2: {Q2}\nQ3: {Q3}\nmin: {train_scores.min():.2f}\nmax: {train_scores.max():.2f}"
    print(stats)
    return stats