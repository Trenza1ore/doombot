# Doom-playing AI via Deep Reinforcement Learning 

**Individual Final Year Project at Cardiff University (BSc)**

---

Utilizing DQN, DRQN and Prioritized Experience Replay to train an agent for playing Doom. 
Scenarios tested: 
- **Deathmatch (modified):** a modified version of deathmatch scenario with different map layout and texture where pickups are removed and killing enemies restore health/armor/ammo
- **Deadly Corridor (modified):** a more Doom-ish version of deadly corridor scenario where the player starts with a shotgun and no longer takes double damage
- **Deadly Corridor (original):** the classic deadly corridor scenario included in ViZDoom

---

This project was developed using Python 3.10.9 on a laptop with a CUDA-enabled graphics card. 
Only ~1.6GB of video memory is needed according to my testing, so it should be able to run even on entry-level Nvidia graphics cards like MX 150/GT 1030.
Remove all .cuda() function calls if trained using CPU instead.

---

This agent was trained using ViZDoom: https://github.com/Farama-Foundation/ViZDoom

---

**P.S.** If you want to change the episode timeout setting to > 1050 for any scenario at Nightmare difficulty (skill level 5), note that enemies respawn after 30 seconds (1050 ticks) unless the Thing_Remove function is called in your ACS script to remove them or they die of special ways specifically defined in Doom's source code.