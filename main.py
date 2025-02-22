"""import sys

from malware_detector import Detector
from utils import InputProcessor, setup_logger

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the path to the config file.")

    # Get the input path from the command line
    input_path = sys.argv[1]
    sys.argv = [sys.argv[0]]  # reset sys.argv
    # Read the input file
    input_processor = InputProcessor(input_path)
    config, label_data = input_processor.run()
    # Setup the logger
    logger = setup_logger(config.path.log)
    logger.info("Logger is set up.")

    # Initialize the MachineUnlearning class
    malware_detector = Detector(config, label_data, logger)
    logger.info("Program started.")
    # Check the command line arguments
    if config.train == True:
        malware_detector.train()
    elif config.predict == True:
        malware_detector.predict()
    elif config.classify == True:
        malware_detector.classify()

    logger.info("Program finished.")
"""

from malware_detector import Detector

if __name__ == "__main__":
    malware_detector = Detector()
    malware_detector.preprocess()
    malware_detector.extractFeature()
    malware_detector.vectorize()
    malware_detector.model()
