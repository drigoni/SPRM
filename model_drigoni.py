import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class MATnet(nn.Module):
	def __init__(self, wordvec, args):
		"""
		:param wordvec: embeddings
		:param args: args from command line
		"""
		super(MATnet, self).__init__()
		# params 
		self.similarity_function = nn.CosineSimilarity(dim=-1)
		self.emb_dim = args.emb_dim
		self.feature_dim = args.feature_dim
		self.cosine_similarity_strategy = args.cosine_similarity_strategy

		# other NN
		self.wordemb = wordvec
		self.indexer = wordvec.word_indexer
		self.wv = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze = False)

		# NN image branch
		self.linear_img = nn.Linear(self.feature_dim+5, self.emb_dim)
		self.linear_img_act = nn.ReLU()
		nn.init.xavier_uniform_(self.linear_img.weight)
		nn.init.zeros_(self.linear_img.bias)
		# NN text branch
		self.queries_rnn = nn.LSTM(300, self.emb_dim, num_layers=1, bidirectional=False, batch_first=False)

	def forward(self, query, head, label, proposals_features, attrs, bboxes, num_query, num_proposal):
		"""
		:param idx:
		:param query: [B, all_query=32, Q=12] pad with 0
		:param head: [B, all_query=32] pad with 0
		:param label: [B, K=64] pad with 0
		:param proposals_features: [B, K, feature_dim] pad with 0
		:param attrs: [B, K=64] pad with 0
		:param bboxes: [B, K, 5] pad with 0
		:param num_query: number of queries
		:param num_proposal: number of proposals
		:return: prediction_scores, prediction_loss, target 
		"""
		# params
		batch_size = query.shape[0]
		n_queries = query.shape[1]
		n_proposals = proposals_features.shape[1]
		# build variables
		bool_words = torch.greater(query, 0).type(torch.long)	# [B, query, words]
		bool_queries = torch.any(bool_words, dim=-1).type(torch.long)	# [B, query]
		num_words = torch.sum(bool_words, dim=-1)				# [B, query]
		bool_proposals = torch.greater(bboxes, 0).type(torch.long)	# [B, proposals, 5]
		bool_proposals = torch.any(bool_proposals, dim=-1).type(torch.long)	# [B, proposals]
		num_proposal = torch.sum(bool_proposals, dim=-1)				# [B]

		# generate boolean mask
		mask_bool_queries_ext =  bool_queries.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, batch_size, n_proposals)
		mask_bool_proposals_ext =  bool_proposals.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_queries, 1, 1)
		mask = (mask_bool_queries_ext * mask_bool_proposals_ext) == 0	# [b, n_queries, b, n_proposal]

		# get embeddings
		q_emb, k_emb, attr_emb, h_emb = self._encode(query, label, attrs, head)
		v_feat = self._get_image_features(bboxes, proposals_features, bool_proposals)
		q_feat = self._get_query_features(h_emb, num_words)

		# get similarity scores
		# NOTE: everything from here is masked with -100 and not 0.
		concepts_similarity = self._get_concept_similarity(q_emb, k_emb, num_words, mask) 	# TODO: change q_emb with h_emb in _encode
		prediction_scores = self._get_predictions(q_feat, v_feat, concepts_similarity, mask)
		prediction_loss, target = self.get_predictions_for_loss(prediction_scores, bool_queries)
		return prediction_scores, prediction_loss, target
	
	def predict(self, query, head, label, feature, attrs, num_query, num_proposal, bboxes):
		prediction_scores, prediction_loss, target = self.forward(query, head, label, feature, attrs, bboxes, num_query, num_proposal)
		batch_size = prediction_scores.shape[0]
		n_query = prediction_scores.shape[1]
		n_proposal = prediction_scores.shape[3]
		device = torch.device("cuda:0" if self.linear_img.weight.is_cuda else "cpu")

		# select the best proposal
		index = torch.arange(batch_size, device=device)  # [b]
		index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, n_query, 1, n_proposal)     # [b, n_query, 1, n_proposal]
		predictions = torch.gather(prediction_scores, 2, index).squeeze(2) 	# [b, n_query, n_proposal]
		prediction_labels = torch.argmax(predictions, dim = -1)  # [B, n_query]

		bbox_ext = bboxes.unsqueeze(1).repeat(1, n_query, 1, 1)	# [B, n_query, n_proposals, 5]
		bbox_ext = bbox_ext[..., :4] # remove area in last position # [B, n_query, n_proposals, 4]
		index_bbox =  prediction_labels.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 4)	# [B, n_query, 1, 4]
		selected_bbox = torch.gather(bbox_ext, 2, index_bbox).squeeze(2)

		return predictions, selected_bbox, prediction_loss, target
	
	def get_predictions_for_loss(self, prediction_scores, bool_queries):
		"""
		:param prediction_scores: [b, query, b, proposal] masked with value=-100
		:param bool_queries: mask for the length of the queries [b, query]
		:return scores: score to use in the loss [b, b]
		:return target: the target values to use during the loss calculation [b]
		"""
		device = torch.device("cuda:0" if self.linear_img.weight.is_cuda else "cpu")
		batch_size = prediction_scores.shape[0]
		num_queries = bool_queries.sum(dim=-1)		# [b]
		
		scores, _ = torch.max(prediction_scores, dim=-1) # [b, query, b]
		scores = torch.sum(scores, dim=1) / num_queries.unsqueeze(-1)   # [b, b]

		target = torch.eye(batch_size, device=device)	# [b, b]
		target = torch.argmax(target, dim=-1)	# [b]
		return scores, target

	def _get_predictions(self, q_feat, v_feat, concepts_similarity, mask, weight=0.5):
		"""
		:param q_feat: features of the queries [b, query, dim]
		:param v_feat: visual features of the bounding boxes [b, proposal, dim]
		:param concepts_similarity: concept similarity scores [b, query, b, proposal]
		:param mask: boolean mask [b, query, b, proposal] with values as {True, False}
		:return predictions: [b, query, b, proposal]
		"""
		# params
		batch_size = q_feat.shape[0]
		n_queries = q_feat.shape[1]
		n_proposals = v_feat.shape[1]

		# calculate similarity scores
		q_feat_ext = q_feat.unsqueeze(2).unsqueeze(2) #.repeat(1, 1, batch_size, n_proposals, 1)		# [b, query, b, proposal, dim]
		k_feat_ext = v_feat.unsqueeze(0).unsqueeze(0) #.repeat(batch_size, n_queries, 1, 1, 1)		# [b, query, b, proposal, dim]
		predictions_qk = self.similarity_function(q_feat_ext, k_feat_ext)		# [b, query, b, proposal]

		# merge the predictions with the concept similarity scores
		predictions = weight*predictions_qk + (1-weight)*concepts_similarity
		
		# mask
		predictions = predictions.masked_fill(mask, -100)

		return predictions

	def _get_concept_similarity(self, q_emb, k_emb, num_words, mask):
		"""
		:param h_emb: embedding of the queries words heads [b, query, words, dim]
		:param k_emb: embedding of the bounding boxes classes [b, proposal, dim]
		:param num_words: number of words for each query [b, query]
		:param mask: boolean mask [b, query, b, proposal] with values as {True, False}
		:return scores: predicted scores [b, query, b, proposal]
		"""
		batch_size = q_emb.shape[0]
		n_queries = q_emb.shape[1]
		n_words = q_emb.shape[2]
		n_proposals = k_emb.shape[1]

		# calculate similarity scores
		with torch.no_grad():
			if self.cosine_similarity_strategy == 'mean':
				q_emb = torch.sum(q_emb, dim=2) / num_words.unsqueeze(-1)		# [b, query, dim]
				q_emb_ext = q_emb.unsqueeze(2).unsqueeze(2)	# [b, query, b, 1, 1]
				k_emb_ext = k_emb.unsqueeze(0).unsqueeze(0)	# [1, 1, b, proposal, dim]
				scores = self.similarity_function(q_emb_ext, k_emb_ext)		# [b, query, b, proposal]
			elif self.cosine_similarity_strategy == 'max':
				# TODO: finisci
				# select the score considering only the most similar words in a query with the labels
				q_emb_ext = q_emb.unsqueeze(3).unsqueeze(3)	# [b, query, 1, 1, dim]
				k_emb_ext = k_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)	# [1, 1, 1, b, proposal, dim]
				scores_middle = self.similarity_function(q_emb_ext, k_emb_ext)	# [b, query, word, b, proposal]
				scores_prop_max, scores_prop_idx =  torch.max(scores_middle, dim=2)	# [b, query, b, proposals]
				scores = scores_prop_max
			else:
				print("Error, cosine_similarity_strategy '{}' not defined. ".format(self.cosine_similarity_strategy))
				exit(1)
			
			# mask
			scores = scores.masked_fill(mask, -100)

		return scores

	def _get_query_features(self, q_emb, q_length, norm=1, emb_dim=300):
		"""
		:param: q_emb: embedding of each phrase
		:param: q_length: length of each phrase
		:return: tensor of embedding features for each query
		"""
		device = torch.device("cuda:0" if self.linear_img.weight.is_cuda else "cpu")
		batch_size = q_emb.size()[0]
		n_queries = q_emb.size()[1]
		mask = q_length.unsqueeze(-1).repeat(1, 1, emb_dim).eq(0)	# [b, query, emb_dim]

		q_emb = q_emb.view(-1, q_emb.size()[-2], q_emb.size()[-1])
		q_emb = q_emb.permute(1, 0, 2).contiguous()
		# NOTE: we need to fix the bug about queries with lengths 0. On cpu required by torch
		q_length_clamp = q_length.view(-1).clamp(min=1).cpu()
		queries_pack_emb = rnn.pack_padded_sequence(q_emb, q_length_clamp, enforce_sorted=False)
		queries_x_o, (queries_x_h, queries_x_c) = self.queries_rnn(queries_pack_emb)
		queries_x_o = rnn.pad_packed_sequence(queries_x_o, batch_first=False)                    # (values, length)
		# due to padding we need to get indexes in this way. On device now.
		idx = (queries_x_o[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, queries_x_o[0].size()[2]).to(device)
		queries_x = torch.gather(queries_x_o[0], 0, idx)                                        # [1, b*n_queries, 2048]
		queries_x = queries_x.permute(1, 0, 2).contiguous().unsqueeze(0)                        # [b*n_queries, 2048]
		# back to batch size dimension
		queries_x = queries_x.squeeze(1).view(batch_size, n_queries, emb_dim)              # [b, n_queries, dim]
		queries_x = queries_x.masked_fill(mask, 0) 
		# normalize features
		queries_x_norm = F.normalize(queries_x, p=norm, dim=-1)
		return queries_x_norm

	def _get_image_features(self, boxes, boxes_feat, bool_proposals, norm=1):
		"""
		Normalize bounding box features and concatenate its spacial features (position and area).

		:param boxes: A [b, proposal, 5] tensor
		:param boxes_feat: A [b, proposal, fi] tensor
		:param num_proposal: A [b] tensor
		:return: A [b, proposal, fi + 5] tensor
		"""
		mask = bool_proposals.unsqueeze(-1)	# [b, proposals, 1]
		boxes_feat = F.normalize(boxes_feat, p=norm, dim=-1)
		boxes_feat = torch.cat([boxes_feat, boxes], dim=-1)		# here there is also the area
		boxes_feat = self.linear_img_act(self.linear_img(boxes_feat))
		boxes_feat = boxes_feat.masked_fill(mask==0, 0)
		return boxes_feat

	def _encode(self, query, label, attrs, head):
		"""
		:param query: query phrases [B, queries, words]
		:param label: object labels, predicted by the detector [B, objects]
		:param attrs: object attributes, predicted by the detector [B, objects]
		:param head: query phrases head [B, heads]

		:return: 	q_emb[B, queries, words, dim] for query word embedding;
					k_emb[B, objects, dim] for object embedding
					attr_emb[B, objects, dim] for attribute embedding
					head_emb[B, heads, dim] for head word embedding;
		"""

		q_emb = self.wv(query)
		k_emb = self.wv(label)
		attr_emb = self.wv(attrs)
		head_emb = self.wv(query) # TODO: risolvi
		return q_emb, k_emb, attr_emb, head_emb

