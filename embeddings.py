import numpy as np 

def readBaroni(path):
	embeddings = {}
	file = open(path,'r', encoding="utf8")

	for line in file:
		tokens = line.split('\t')

		#since each line's last token content '\n'
		# we need to remove that
		tokens[-1] = tokens[-1].strip()

		#each line has 400 tokens
		for i in range(1, len(tokens)):
			tokens[i] = float(tokens[i])

		embeddings[tokens[0]] = tokens[1:-1]

	return embeddings