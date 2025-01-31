from sacrebleu import corpus_bleu
import typing as t

class BlueScore():
    """
    A class to compute BLEU scores for text evaluation.
    Attributes:
        references (List[List[str]]): A list of reference texts.
    """
    def __init__(self,references:t.List[str]) -> None:
        """
        Initializes the BlueScore class with the specified references.
        Args:
            references (List[str]): A list of reference texts.
        """
        self.references = [[ref] for ref in references]
    
    def _getScore(self,prediction:str) -> float:
        """
        Computes the BLEU score for the given prediction against the references.
        Args:
            prediction (str): The predicted text.
        Returns:
            float: The computed BLEU score.
        """
        return corpus_bleu([prediction],self.references).score/100