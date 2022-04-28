from transformers import set_seed
from myutils import Configs

def main():
    config = Configs()
    config.parse_from_argv()

if __name__ == "__main__":
    main()
