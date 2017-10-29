# -*- coding:utf-8 -*-
import codecs
from collections import Counter, defaultdict
from tqdm import tqdm
import operator

if __name__ == '__main__':
	datafile = 'two_1_2.0_100w.data'
	result = defaultdict(int)
	count = 0
	with codecs.open(datafile, 'r', 'utf-8') as f:
		for line in tqdm(f):
			line = line.strip('\n')
			temp = Counter(line)
			if not line or '[' not in line or temp['['] != temp[']']:
				continue
			count += 1
			while line and '[' and ']' in line:
				try:
					right = line.split('[', 1)[1]
					ans, line = right.split(']', 1)
					result[ans] += 1
				except:
					break
	result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
	print('MSG : Done for scanning!')
	print(count)
	with codecs.open('answer.data', 'w', 'utf-8') as wf:
		for key, value in result:
			if value > 280:
				wf.write(key)
				wf.write('\n')
			else:
				break
	print('Done!')


