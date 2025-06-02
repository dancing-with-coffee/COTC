import argparse
import yaml

###

class Config(object):
    def __init__(self, config_path, version):
        super().__init__()

        # config from file, default setting
        with open(config_path, mode="r", encoding="utf-8") as config_stream:
            config_dictionary = {"version": version, **yaml.safe_load(config_stream)}

        # config from command, modified setting
        parser = argparse.ArgumentParser()

        for key, value in config_dictionary.items():
            cast = type(value)

            if cast == bool: # special treatment for bool
                cast = self.bool

            parser.add_argument("--" + key, default=value, type=cast)

        parser_dictionary = parser.parse_args().__dict__

        # config loading
        self.__dict__.update(config_dictionary)
        self.__dict__.update(parser_dictionary)

    def str(self):
        detail = "# # # # # # # # # # # # # # # # # # # # # # # # # # # #\n"

        for key, value in self.__dict__.items():
            detail = detail + ("#%25s : %-25s#\n" % (key, value))

        detail = detail + "# # # # # # # # # # # # # # # # # # # # # # # # # # # #"

        return detail

    def bool(self, value):
        if value == "True":
            return True
        elif value == "False":
            return False
        else:
            raise argparse.ArgumentTypeError("invalid bool value: '" + value + "'")

###
