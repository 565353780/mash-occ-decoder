from mash_occ_decoder.Demo.trainer import demo as demo_train
from mash_occ_decoder.Demo.detector import (
    demo_file as demo_detect_file,
    demo_folder as demo_detect_folder
)

if __name__ == "__main__":
    demo_train()
    demo_detect_file()
    demo_detect_folder()
