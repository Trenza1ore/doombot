import numpy as np
from numpy.random import choice, randint, default_rng

class ReplayMemory:
    def __init__(self, res: tuple[int, int]=(240, 320), ch_num: int=3, size: int=40000, 
                 dtypes: list[object]=[np.uint8, np.float32]):
        """A class that implements replay memory for experience replay and prioritized experience replay.

        Args:
            res (tuple[int, int], optional): resolution in form of (height, width). Defaults to (240, 320).
            ch_num (int): number of colour channels. Defaults to 3.
            size (int): maximum size, starts deleting old memories after maximum reached. Defaults to 40000.
            dtypes (list[object]): data type of frame, reward and (if any) features, datatype for features 
            should be passed in as strings like "np.uint8" or "bool". Defaults to [np.uint8, np.float32].
        """        
        self.max_size = size
        self.max_index = size - 1
        self.ch_num = ch_num
        self.dtype = {
            "frame"     : (dtypes[0], (size, ch_num, *res)),
            "reward"    : (dtypes[1], (size, ))
        }
        
        if len(dtypes > 2):
            self.dtype["features"] = (*dtypes[2:], )
            self.feature_num = len(self.dtype["features"])
            for i in range(self.feature_num):
                exec(f"self.feature_{i} = np.zeros(size, dtype={self.dtype['features'][i]})")
        else:
            self.feature_num = 0
            
        if size < 65_536:
            self.indices = np.arange(size, dtype=np.uint16)
        elif size < 4_294_967_296:
            self.indices = np.arange(size, dtype=np.uint32)
        else:
            self.indices = np.arange(size, dtype=np.uint64)
        
        self.frames = np.zeros((size, ch_num, *res), dtype=dtypes[0])
        self.rewards = np.zeros(size, dtype=dtypes[1])
        self.isTerminate = np.full(size, False)
        
        self.__ptr = -1
        self.rng = default_rng()
        
    def add(self, frame: np.ndarray[np.integer], reward: np.floating):
        self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
        self.frames[self.__ptr, :, :, :] = frame
        self.rewards[self.__ptr] = reward
    
    def bulk_add_unsafe(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], n: int):
        self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
        end = self.__ptr + n
        self.frames[self.__ptr:end, :, :, :] = frame
        self.rewards[self.__ptr:end] = reward
    
    def bulk_add(self, frame: np.ndarray[np.integer], reward: np.ndarray[np.floating], n: int):
        self.__ptr = self.__ptr + 1 if self.__ptr < self.max_index else 0
        if (free_size := (self.max_size - self.__ptr)) < n:
            self.frames[self.__ptr:, :, :, :] = frame[:free_size, :, :, :]
            self.rewards[self.__ptr:] = reward[:free_size]
            remaining = n - free_size
            self.frames[:remaining, :, :, :] = frame[remaining:, :, :, :]
            self.rewards[:remaining] = reward[remaining:]
            self.__ptr = remaining - 1
        else:
            end = self.__ptr + n
            self.frames[self.__ptr:end, :, :, :] = frame
            self.rewards[self.__ptr:end] = reward
            self.__ptr = end - 1
    
    # def replay(self, n: int) -> tuple[np.ndarray[np.integer], np.ndarray[np.floating]]:
    #     random_indices = randint(0, self.max_size, size=n)
    #     return (self.frames[random_indices, :, :, :], self.rewards[random_indices])
    
    # def replay_p(self, n: int, r: bool=True) -> tuple[np.ndarray[np.integer], np.ndarray[np.floating]]:
    #     scores = self.rewards - np.min(self.rewards)
    #     scores = scores / np.sum(scores)
    #     random_indices = choice(self.indices, size=n, replace=r, p=scores)
    #     return (self.frames[random_indices, :, :, :], self.rewards[random_indices])
    
    def replay(self, n: int) -> np.ndarray[np.unsignedinteger]:
        return randint(0, self.max_size, size=n)
    
    def replay_p(self, n: int, r: bool=True) -> np.ndarray[np.unsignedinteger]:
        scores = self.rewards - np.min(self.rewards)
        scores = scores / np.sum(scores)
        return choice(self.indices, size=n, replace=r, p=scores)
        
    def __len__(self) -> int:
        return self.__ptr + 1
    
    def __str__(self) -> str:
        return f"ReplayMemory(f:{self.dtype['frame']}, r:{self.dtype['reward']})"
    
    def __repr__(self) -> str:
        return self.__str__()