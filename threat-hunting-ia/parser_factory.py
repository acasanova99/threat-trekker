from cic_parser import CicParser
from uwf_parser import UwfParser
from ton_parser import TonParser
import logging


class ParserFactory:
    @staticmethod
    def instantiate_parser(path_to_dataset: str):
        """
        Initialize an entailed parser for each of the datasets.
        Args:
            path_to_dataset: Path introduced by the user to get the input informacion.

        Returns: An instance of the concrete parser.
        """
        logger = logging.getLogger('ThreatTrekker')

        if "cic-dataset" in path_to_dataset:
            return CicParser()
        elif "ton-dataset" in path_to_dataset:
            return TonParser()
        elif "uwf-dataset" in path_to_dataset:
            return UwfParser()
        else:
            logger.error(
                f'This version of ThreatTrekker does not provide and implementation for the dataset: {path_to_dataset}')
            raise ValueError("Unknown Dataset")
