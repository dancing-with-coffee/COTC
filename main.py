import sys
sys.dont_write_bytecode = True

###

from util.packet import *
from util.config import Config
from util.logger import Logger

###

if __name__ == "__main__":
    assert len(sys.argv) >= 3
    assert sys.argv[1] == "--version"

    version = sys.argv[2]

    exec("from " + version + ".data import Data")
    exec("from " + version + ".model import Model")
    exec("from " + version + ".runner import Runner")

    config = Config("runner/config/" + version + ".yml", version=version)
    logger = Logger("runner/logger/" + config.logger, version=version)

    data = Data(config, logger)
    model = Model(config, logger)
    runner = Runner(config, logger, data=data, model=model)

    runner.train()

###
