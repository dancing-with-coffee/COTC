import os

###

class Logger(object):
    def __init__(self, logger_path, version):
        super().__init__()

        while os.path.exists(logger_path):
            logger_path = logger_path + "+"

        self.save(logger_path + "/backup", version=version.replace(".", "/"))

        self.logger_path = logger_path + "/log.txt"

    def log(self, string):
        with open(self.logger_path, mode="a", encoding="utf-8") as logger_stream:
            logger_stream.write(string + "\n")

        print(string)

    def save(self, logger_path, version):
        os.system("mkdir -p " + logger_path)
        os.system("cp " + version + "/* " + logger_path)

###
