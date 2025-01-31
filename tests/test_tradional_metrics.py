import unittest
from aiklyra import RougeScore 
from aiklyra import BlueScore

class TestRougeScore(unittest.TestCase):
    def setUp(self):
        self.rouge1_scorer = RougeScore(stemmer=True, score_type=['rouge1'])
        self.rougeL_scorer = RougeScore(stemmer=False, score_type=['rougeL'])

    def test_rouge1_score(self):
        response = "The quick brown fox jumps over the lazy dog"
        context = "The quick brown fox jumps over the lazy dog"
        score = self.rouge1_scorer._getScore(response, context)
        score = self.rougeL_scorer._getScore(response, context)
        print(f"ROUGE-L Score with different texts: {score}")

    def test_rouge1_score_with_partial_match(self):
        response = "The quick brown fox"
        context = "The quick brown fox jumps over the lazy dog"
        score = self.rouge1_scorer._getScore(response, context)
        print(f"ROUGE-1 Score with partial match: {score}")

    def test_rougeL_score_with_partial_match(self):
        response = "The quick brown fox"
        context = "The quick brown fox jumps over the lazy dog"
        score = self.rougeL_scorer._getScore(response, context)
        print(f"ROUGE-L Score with partial match: {score}")

    def test_rouge1_score_with_no_match(self):
        response = "The quick brown fox"
        context = "Lorem ipsum dolor sit amet"
        score = self.rouge1_scorer._getScore(response, context)
        print(f"ROUGE-1 Score with no match: {score}")

    def test_rougeL_score_with_no_match(self):
        response = "The quick brown fox"
        context = "Lorem ipsum dolor sit amet"
        score = self.rougeL_scorer._getScore(response, context)
        print(f"ROUGE-L Score with no match: {score}")

class TestBlueScore(unittest.TestCase):
    def setUp(self):
        self.references = ["The quick brown fox jumps over the lazy dog"]
        self.blue_scorer = BlueScore(references=self.references)

    def test_blue_score(self):
        prediction = "The quick brown fox jumps over the lazy dog"
        score = self.blue_scorer._getScore(prediction)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_blue_score_with_different_text(self):
        prediction = "The quick brown fox jumps over the lazy cat"
        score = self.blue_scorer._getScore(prediction)
        self.assertLess(score, 1.0)

if __name__ == '__main__':
    unittest.main()