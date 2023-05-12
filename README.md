# Doom-playing AI via Deep Reinforcement Learning 

**Individual Final Year Project at Cardiff University (BSc)**
---
Utilizing DQN, DRQN and Prioritized Experience Replay to train an agent for playing Doom. 
Scenarios tested: 
- **Deathmatch (modified):** a modified version of deathmatch scenario with different map layout and texture where pickups are removed and killing enemies restore health/armor/ammo
- **Deadly Corridor (modified):** a modified version of deadly corridor scenario where the player starts with a shotgun
- **Deadly Corridor (original):** the classic deadly corridor scenario
---
This project was developed using Python 3.10.9 on a laptop with CUDA-enabled graphics card. 
Remove all .cuda() function calls if trained using CPU instead.