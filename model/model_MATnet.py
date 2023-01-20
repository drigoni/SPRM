import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from typing import Dict, List, Callable

class MATnet(nn.Module):
	def __init__(self, wordvec, args):
		"""
		:param wordvec: embeddings
		:param args: args from command line
		"""
		super(MATnet, self).__init__()
		# params 
		self.EMB_DIM = args.emb_dim
		self.WORD_EMB_DIM = args.word_emb_dim
		self.IMG_FEATURES_DIM = args.feature_dim

		# other NN
		self.wordemb = wordvec
		self.indexer = wordvec.word_indexer
		self.wv = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze = False)

		# NN image branch
		self.linear_img = nn.Linear(20, 20)
		self.linear_f = nn.Linear(self.IMG_FEATURES_DIM, self.EMB_DIM)
		self.linear_f.weight.data = torch.zeros(self.EMB_DIM, self.IMG_FEATURES_DIM)


		# NN text branch
		self.queries_softmax = nn.Softmax(dim = -1)
		self.linear_p = nn.Linear(self.WORD_EMB_DIM, self.EMB_DIM)
		self.linear_p.weight.data = torch.eye(self.WORD_EMB_DIM)
		self.linear_mini = nn.Linear(self.WORD_EMB_DIM, self.EMB_DIM)
		

	def forward(self, query, head, label, proposals_features, attrs, bboxes):
		"""
		NOTE: PAD is always 0 and UNK is always 1 by construction.
		:param idx:
		:param query: [B, all_query=32, Q=12] pad with 0
		:param head: [B, all_query=32] pad with 0
		:param label: [B, K=64] pad with 0
		:param proposals_features: [B, K, feature_dim] pad with 0
		:param attrs: [B, K=64] pad with 0
		:param bboxes: [B, K, 5] pad with 0
		:return: prediction_scores, prediction_loss, target 
		"""
		# params
		batch_size = query.shape[0]
		n_queries = query.shape[1]
		n_proposals = proposals_features.shape[1]
		# build variables
		bool_words = torch.greater(query, 0).type(torch.long)					# [B, query, words]
		bool_queries = torch.any(bool_words, dim=-1).type(torch.long)			# [B, query]
		num_words = torch.sum(bool_words, dim=-1)								# [B, query]
		bool_proposals = torch.greater(bboxes, 0).type(torch.long)				# [B, proposals, 5]
		bool_proposals = torch.any(bool_proposals, dim=-1).type(torch.long)		# [B, proposals]
		num_proposal = torch.sum(bool_proposals, dim=-1)						# [B]

		# generate boolean mask
		mask_bool_queries_ext =  bool_queries.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, batch_size, n_proposals)
		mask_bool_proposals_ext =  bool_proposals.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_queries, 1, 1)
		mask = (mask_bool_queries_ext * mask_bool_proposals_ext) == 0	# [b, n_queries, b, n_proposal]

		# get embeddings. NOTE: inputs padded with 0
		q_emb, k_emb = self._encode(query, label, attrs, head, bool_words, bool_proposals)
		# get features. NOTE: inputs padded with 0
		v_feat = self._get_image_features(bboxes, proposals_features, k_emb, bool_proposals)
		q_feat = self._get_query_features_att(q_emb, k_emb, bool_queries, bool_proposals) 

		# get similarity scores
		# NOTE: everything from here is masked with -1e8 and not 0.
		prediction_scores = self._get_predictions(q_feat, v_feat, mask)
		prediction_loss, target = self.get_predictions_for_loss(prediction_scores, bool_queries, bool_proposals, mask)

		return prediction_scores, prediction_loss, target
	
	def predict(self, query, head, label, feature, attrs, bboxes):
		prediction_scores, prediction_loss, target = self.forward(query, head, label, feature, attrs, bboxes)
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
		selected_bbox = torch.gather(bbox_ext, 2, index_bbox).squeeze(2)	# [B, n_query, 4]

		return predictions, prediction_loss, selected_bbox
	
	def get_predictions_for_loss(self, prediction_scores, bool_queries, bool_proposals, mask):
		"""
		:param prediction_scores: [b, query, b, proposal] masked with value=0
		:param bool_queries: mask for the length of the queries [b, query]
		:param bool_proposals: mask for queries [b, proposals]
		:param mask: boolean mask [b, query, b, proposal] with values as {True, False}
		:return scores: score to use in the loss [b, b]
		:return target: the target values to use during the loss calculation [b]
		"""
		device = torch.device("cuda:0" if self.linear_img.weight.is_cuda else "cpu")
		batch_size = prediction_scores.shape[0]
		num_queries = bool_queries.sum(dim=-1)		# [b]
		mask_queries = bool_queries.unsqueeze(-1).eq(0)
		
		scores, _ = torch.max(prediction_scores, dim=-1) 					# [b, query, b]

		# now pad scores with 0 and not with -1e8 in order to do the mean
		scores = scores.masked_fill(mask_queries, 0) 
		scores = scores.sum(dim=1) / num_queries.unsqueeze(-1)   # [b, b]

		target = torch.eye(batch_size, device=device)	# [b, b]
		target = torch.argmax(target, dim=-1)	# [b]
		return scores, target

	def _get_predictions(self, q_feat, v_feat, mask):
		"""
		:param q_feat: features of the queries [b, query, dim]. pad with 0
		:param v_feat: visual features of the bounding boxes [b, proposal, dim]. pad with 0
		:param mask: boolean mask [b, query, b, proposal] with values as {True, False}
		:return predictions: [b, query, b, proposal]
		"""
		# attention sulle phrases
		predictions = torch.einsum('avd, bqd -> bqav', v_feat, q_feat)  # [B1, K, dim] x [B2, querys, dim] => [B2, querys, B1, K]
		# mask
		predictions = predictions.masked_fill(mask, 0)
		return predictions


	def _get_query_features_att(self, q_emb, k_emb, bool_queries, bool_proposals):
		"""
		:param: q_emb: embedding of each phrase [B, querys, words, dim]. pad with 0
		:param: k_emb: embedding of box labels [B, proposals, dim]. pad qith 0
		:param bool_queries: mask for queries [b, query]
		:param bool_proposals: mask for queries [b, proposals]
		:return: query embedding
		"""
		# generate masks
		mask_queries = bool_queries.unsqueeze(-1).eq(0)						# [B, queries, 1]
		mask_proposals = bool_proposals.unsqueeze(1).unsqueeze(1).eq(0)		# [B, 1, 1, proposals]


		# q_emb [B, querys, Q, dim]
		scale = 1.0 / np.sqrt(k_emb.size(-1))
		att = torch.einsum('byqd,bkd ->byqk', q_emb, k_emb)		# [B, queries, words, proposals]
		# mask before softmax 
		att = att.masked_fill(mask_proposals, -1e8)
		att = self.queries_softmax(att.mul_(scale))  			# [B, querys, words, proposals] Pad with 0

		q_max_att = torch.max(att, dim=3).values  				# [B, querys, words]
		# mask before softmax
		q_max_att = q_max_att.masked_fill(mask_queries, -1e8)	
		q_max_norm_att = self.queries_softmax(q_max_att)		# [B, querys, words] Pad with 0
		# attended
		p_emb = torch.einsum('byq,byqd -> byd', q_max_norm_att, q_emb)  # [B, querys, dim]
		p_emb = self.linear_p(p_emb) + 1e-5 * self.linear_mini(p_emb)
		
		# mask
		p_emb = p_emb.masked_fill(mask_queries, 0)

		return p_emb

	def _get_image_features(self, boxes, boxes_feat, k_emb, bool_proposals):
		"""
		Normalize bounding box features and concatenate its spacial features (position and area).

		:param boxes: A [b, proposal, 5] tensor
		:param boxes_feat: A [b, proposal, fi] tensor
		:param k_emb: A [b, proposals, dim] tensor
		:param bool_proposals: A [b, proposals] tensor
		"""
		mask = bool_proposals.unsqueeze(-1).eq(0)	# [b, proposals, 1]
		boxes_feat = self.linear_f(boxes_feat)
		boxes_feat = boxes_feat + k_emb
		boxes_feat = boxes_feat.masked_fill(mask, 0)
		return boxes_feat

	def _encode(self, query, label, attrs, head, bool_words, bool_proposals):
		"""
		:param query: query phrases [B, queries, words]
		:param label: object labels, predicted by the detector [B, objects]
		:param attrs: object attributes, predicted by the detector [B, objects]
		:param head: query phrases head [B, heads]
		:param bool_words: logic mask for words [B, query, words]
		:param bool_proposals: logic mask for proposals [B, proposals]
		:return: 	q_emb[B, queries, words, dim] for query word embedding;
					k_emb[B, objects, dim] for object embedding
					attr_emb[B, objects, dim] for attribute embedding
					head_emb[B, heads, dim] for head word embedding;
		"""
		# params
		mask_words = bool_words.unsqueeze(-1).eq(0)
		mask_proposals = bool_proposals.unsqueeze(-1).eq(0)

		q_emb = self.wv(query)
		k_emb = self.wv(label)
		# attr_emb = self.wv(attrs)
		# head_emb = self.wv(query)

		# mask
		q_emb = q_emb.masked_fill(mask_words, 0)
		k_emb = k_emb.masked_fill(mask_proposals, 0)

		return q_emb, k_emb