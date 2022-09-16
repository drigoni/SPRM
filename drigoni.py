import pickle
import os


def check_ids():
	train_imgid2idx = pickle.load(open("data/flickr30k/train_imgid2idx.pkl", "rb"))
	test_imgid2idx = pickle.load(open("data/flickr30k/test_imgid2idx.pkl", "rb"))
	val_imgid2idx = pickle.load(open("data/flickr30k/val_imgid2idx.pkl", "rb"))

	with open("data/flickr30k/train.txt", 'r') as f:
		real_train_imgid2idx = f.read()[:-1].split('\n')
	with open("data/flickr30k/test.txt", 'r') as f:
		real_test_imgid2idx = f.read()[:-1].split('\n')
	with open("data/flickr30k/val.txt", 'r') as f:
		real_val_imgid2idx = f.read()[:-1].split('\n')


	assert(len(real_train_imgid2idx) == len(train_imgid2idx.keys()))
	assert(len(real_test_imgid2idx) == len(test_imgid2idx.keys()))
	assert(len(real_val_imgid2idx) == len(val_imgid2idx.keys()))

	# print(list(zip(real_val_imgid2idx, val_imgid2idx.keys())))

	for id in train_imgid2idx.keys():
		assert str(id) in real_train_imgid2idx
	for id in test_imgid2idx.keys():
		assert str(id) in real_test_imgid2idx
	for id in val_imgid2idx.keys():
		assert str(id) in real_val_imgid2idx

	print("done")


if __name__ == "__main__":
	check_ids()