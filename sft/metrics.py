from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sf = SmoothingFunction().method1

class BleuCalculator:
	@staticmethod
	def calculate(pred: str, label: str):
		references = [label.split()]
		candidate = pred.split()
		return sentence_bleu(references, candidate, smoothing_function=sf)

def test_BleuCalculator():
	pred = "select * from a where b = c"
	label = "select * from b where a = c"
	print(BleuCalculator.calculate(pred, label))


