import subprocess
import sys
from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from .dqn_agent import DQNAgent as DQNAgent
    from .nfsp_agent import NFSPAgent as NFSPAgent

from .cfr_agent import CFRAgent
from .human_agents.limit_holdem_human_agent import HumanAgent as LimitholdemHumanAgent
from .human_agents.nolimit_holdem_human_agent import HumanAgent as NolimitholdemHumanAgent
from .random_agent import RandomAgent

# from .rl_agent import RLAgent
