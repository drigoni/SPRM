import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from typing import Dict, List, Callable

class ConceptNet(nn.Module):
	def __init__(self, wordvec, args):
		"""
		:param wordvec: embeddings
		:param args: args from command line
		"""
		super(ConceptNet, self).__init__()
		# params 
		self.EMB_DIM = args.emb_dim
		self.WORD_EMB_DIM = args.word_emb_dim
		self.IMG_FEATURES_DIM = args.feature_dim
		self.COSINE_SIMILARITY_STRATEGY = args.cosine_similarity_strategy	
		self.PREDICTION_WEIGHT = args.cosine_weight
		self.USE_ATT_FOR_QUERY = args.use_att_for_query
		self.USE_MEAN_IN_LOSS = args.use_mean_in_loss
		self.SIMILARITY_STRATEGY = args.similarity_strategy
		self.USE_HEAD_FOR_QUERY_EMBEDDING = args.use_head_for_query_embedding
		self.IMAGE_NET_DROPOUT = args.image_net_dropout
		self.QUERY_NET_DROPOUT = args.query_net_dropout
		self.USE_BIDIRECTIONAL_LSTM = args.use_bidirectional_lstm
		self.LSTM_NUM_LAYERS = args.lstm_num_layers
		self.USE_HEAD_FOR_CONCEPT_EMBEDDING = args.use_head_for_concept_embedding

		# other NN
		self.wordemb = wordvec
		self.indexer = wordvec.word_indexer
		self.wv = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze = False)
		self.wv_freezed = nn.Embedding.from_pretrained(torch.from_numpy(wordvec.vectors), freeze = True)

		# NN image branch
		self.linear_img = nn.Linear(20, 20)
		self.img_mlp = MLP(self.IMG_FEATURES_DIM + 5, self.EMB_DIM, [], F.leaky_relu, dropout=self.IMAGE_NET_DROPOUT)
		# NN text branch
		if self.USE_ATT_FOR_QUERY is False:
			self.queries_rnn = nn.LSTM(self.WORD_EMB_DIM, self.EMB_DIM, num_layers=self.LSTM_NUM_LAYERS, bidirectional=self.USE_BIDIRECTIONAL_LSTM, batch_first=False)
		else:
			self.queries_mlp = MLP(self.WORD_EMB_DIM, self.EMB_DIM, [self.EMB_DIM], F.leaky_relu, dropout=self.QUERY_NET_DROPOUT)
			self.queries_softmax = nn.Softmax(dim = -1)

		if self.SIMILARITY_STRATEGY == 'euclidean_distance':
			self.similarity_function = nn.PairwiseDistance()
		else:
			self.similarity_function = nn.CosineSimilarity(dim=-1)
		

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
		bool_words = torch.greater(query, 0).type(torch.long)							# [B, query, words]
		bool_heads = torch.greater(head, 0).type(torch.long)							# [B, query, words]
		bool_queries = torch.any(bool_words, dim=-1).type(torch.long)			# [B, query]
		num_words = torch.sum(bool_words, dim=-1)													# [B, query]
		num_queries = torch.sum(bool_queries, dim=-1)											# [B]
		bool_proposals = torch.greater(bboxes, 0).type(torch.long)				# [B, proposals, 5]
		bool_proposals = torch.any(bool_proposals, dim=-1).type(torch.long)		# [B, proposals]
		num_proposal = torch.sum(bool_proposals, dim=-1)						# [B]

		# generate boolean mask
		mask_bool_queries_ext =  bool_queries.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, batch_size, n_proposals)
		mask_bool_proposals_ext =  bool_proposals.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_queries, 1, 1)
		mask = (mask_bool_queries_ext * mask_bool_proposals_ext) == 0	# [b, n_queries, b, n_proposal]

		# get embeddings. NOTE: inputs padded with 0
		q_emb, k_emb = self._encode(query, label, attrs, head, bool_words, bool_proposals)
		q_emb_freezed, k_emb_freezed, head_emb_freezed = self._encode_freezed(query, label, attrs, head, bool_words, bool_proposals)
		# get features. NOTE: inputs padded with 0
		v_feat = self._get_image_features(bboxes, proposals_features, k_emb, bool_proposals, 1)
		if self.USE_ATT_FOR_QUERY is False:
			# so use LSTM
			q_feat = self._get_query_features(q_emb, num_words, self.EMB_DIM, 1)
		else:
			q_feat = self._get_query_features_att(q_emb, k_emb, bool_queries, bool_proposals) 

		# get similarity scores
		# NOTE: everything from here is masked with -1e8 and not 0.
		if self.USE_HEAD_FOR_CONCEPT_EMBEDDING:
			new_q_emb = head_emb_freezed
			new_bool_words = bool_heads
			new_bool_queries = torch.any(new_bool_words, dim=-1).type(torch.long)			# [B, query]
			new_num_words = torch.sum(new_bool_words, dim=-1)

			new_mask_bool_queries_ext =  new_bool_queries.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, batch_size, n_proposals)
			new_mask = (new_mask_bool_queries_ext * mask_bool_proposals_ext) == 0	# [b, n_queries, b, n_proposal]

			concepts_similarity = self._get_concept_similarity(new_q_emb, k_emb_freezed, new_num_words, new_mask)
		else:
			concepts_similarity = self._get_concept_similarity(q_emb_freezed, k_emb_freezed, num_words, mask)

		prediction_scores = self._get_predictions(q_feat, v_feat, concepts_similarity, mask, self.PREDICTION_WEIGHT)

		# get query similarity
		if self.USE_HEAD_FOR_QUERY_EMBEDDING:
			new_q_emb = head_emb_freezed
			new_bool_words = bool_heads
		else:
			new_q_emb = q_emb_freezed
			new_bool_words = bool_words
		new_bool_queries = torch.any(new_bool_words, dim=-1).type(torch.long)  # [B, query]
		query_similarity = self._get_query_similarity(new_q_emb, new_bool_words, new_bool_queries)  # [b, b]

		# attention sulle phrases
		# attmap = torch.einsum('avd, bqd -> baqv', k_emb, p_emb)  # [B1, K, dim] x [B2, querys, dim] => [B2, B1, querys, K]
		# attmap_sm = self.softmax(attmap)  # [B2, B1, querys, K]
		# att_obj_sum = torch.sum(attmap_sm, dim = -2)  # [B2, B1, K]
		# maxatt, _ = attmap.max(dim = -1)  # [B1, B2, querys]: B1th sentence to B2th image
		# logits = torch.sum(maxatt, dim = -1).div(num_query.unsqueeze(1).expand(maxatt.size(0), maxatt.size(1)))  # [B1, B2]: B1th sentence to B2th image

		prediction_loss, target = self.get_predictions_for_loss(prediction_scores, bool_queries, bool_proposals, mask)

		return prediction_scores, prediction_loss, target, query_similarity
	
	def predict(self, query, head, label, feature, attrs, bboxes):
		prediction_scores, prediction_loss, target, query_similarity = self.forward(query, head, label, feature, attrs, bboxes)
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

		return predictions, prediction_loss, selected_bbox, target, query_similarity
	
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
		
		if self.USE_MEAN_IN_LOSS is False:
			scores, _ = torch.max(prediction_scores, dim=-1) 					# [b, query, b]
		else:
			# we need to mask negatives values otherwise it make learning pushing in the opposite direction each point that should not be related.
			# scores = prediction_scores.clamp(min=0)								# [b, query, b, proposals]
			scores = prediction_scores												# [b, query, b, proposals]
			# mask boxes
			mask_proposals = bool_proposals.unsqueeze(0).unsqueeze(0).eq(0)		# [1, 1, b, proposals]
			scores = scores.masked_fill(mask_proposals, 0)						# [b, query, b, proposals]
			# at this point, everything to ignore is at 0.
			num_proposals_to_consider = scores.greater(0).type(torch.long).sum(-1)	# [b, query, b]
			scores = scores.sum(dim=-1) / (num_proposals_to_consider + 1e-8) 				# [b, query, b]

		# now pad scores with 0 and not with -1e8 in order to do the mean
		scores = scores.masked_fill(mask_queries, 0) 
		scores = scores.sum(dim=1) / num_queries.unsqueeze(-1)   # [b, b]

		target = torch.eye(batch_size, device=device)	# [b, b]
		target = torch.argmax(target, dim=-1)	# [b]
		return scores, target

	def _get_predictions(self, q_feat, v_feat, concepts_similarity, mask, weight):
		"""
		:param q_feat: features of the queries [b, query, dim]. pad with 0
		:param v_feat: visual features of the bounding boxes [b, proposal, dim]. pad with 0
		:param concepts_similarity: concept similarity scores [b, query, b, proposal]. pad with -1e8
		:param mask: boolean mask [b, query, b, proposal] with values as {True, False}
		:return predictions: [b, query, b, proposal]
		"""
		# params
		batch_size = q_feat.shape[0]
		n_queries = q_feat.shape[1]
		n_proposals = v_feat.shape[1]

		# calculate similarity scores
		q_feat_ext = q_feat.unsqueeze(2).unsqueeze(2).repeat(1, 1, batch_size, n_proposals, 1)	# [b, query, b, proposal, dim]
		v_feat_ext = v_feat.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_queries, 1, 1, 1)	# [b, query, b, proposal, dim]

		# calculate similarity
		predictions_qk = self.similarity_function(q_feat_ext, v_feat_ext)		# [b, query, b, proposal]
		predictions_qk = predictions_qk.masked_fill(mask, -1e8)					# [b, query, b, proposal] pad with -1e8

		# merge the predictions with the concept similarity scores
		predictions = weight*F.softmax(predictions_qk, dim=-1) + (1-weight)*F.softmax(concepts_similarity, dim=-1)
		# predictions = concepts_similarity

		# mask
		predictions = predictions.masked_fill(mask, 0)

		return predictions

	def _get_concept_similarity(self, q_emb, k_emb, num_words, mask):
		"""
		:param q_emb: embedding of the queries words heads [b, query, words, dim]. pad with 0
		:param k_emb: embedding of the bounding boxes classes [b, proposal, dim]. pad with 0
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
			if self.COSINE_SIMILARITY_STRATEGY == 'mean':
				q_emb = torch.sum(q_emb, dim=2) / num_words.unsqueeze(-1)		# [b, query, dim]
				q_emb_ext = q_emb.unsqueeze(2).unsqueeze(2)	# [b, query, 1, 1, dim]
				k_emb_ext = k_emb.unsqueeze(0).unsqueeze(0)	# [1, 1, b, proposal, dim]
				scores = self.similarity_function(q_emb_ext, k_emb_ext)		# [b, query, b, proposal]
				# mask
				scores = scores.masked_fill(mask, -1e8)
			elif self.COSINE_SIMILARITY_STRATEGY == 'max':
				# select the score considering only the most similar words in a query with the labels
				q_emb_ext = q_emb.unsqueeze(3).unsqueeze(3)	# [b, query, 1, 1, dim]
				k_emb_ext = k_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)	# [1, 1, 1, b, proposal, dim]
				scores_middle = self.similarity_function(q_emb_ext, k_emb_ext)	# [b, query, word, b, proposal]
				# mask before selecting max values
				scores_middle = scores_middle.masked_fill(mask.unsqueeze(2), -1e8)
				# select best values
				scores_prop_max, scores_prop_idx =  torch.max(scores_middle, dim=-1)	# [b, query, words, b]
				scores_words_max, scores_words_idx =  torch.max(scores_prop_max, dim=2)	# [b, query, b]
				index_best_word = scores_words_idx.unsqueeze(2).unsqueeze(-1)	# [b, query, 1, b, 1]
				scores = torch.gather(scores_middle, 2, index_best_word).squeeze(2)	# [b, query, b, proposal]
			else:
				print("Error, cosine_similarity_strategy '{}' not defined. ".format(self.COSINE_SIMILARITY_STRATEGY))
				exit(1)
			
		return scores

	def _get_query_features(self, q_emb, q_length, emb_dim, norm):
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

		if self.training:
			queries_x = nn.Dropout(p=self.QUERY_NET_DROPOUT)(queries_x)
		# normalize features
		# queries_x_norm = F.normalize(queries_x, p=norm, dim=-1)
		return queries_x

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
		p_emb = self.queries_mlp(p_emb)									# [B, querys, dim]
		
		# mask
		p_emb = p_emb.masked_fill(mask_queries, 0)

		return p_emb

	def _get_image_features(self, boxes, boxes_feat, k_emb, bool_proposals,  norm):
		"""
		Normalize bounding box features and concatenate its spacial features (position and area).

		:param boxes: A [b, proposal, 5] tensor
		:param boxes_feat: A [b, proposal, fi] tensor
		:param k_emb: A [b, proposals, dim] tensor
		:param bool_proposals: A [b, proposals] tensor
		:return boxes_feat: A [b, proposal, fi + 5] tensor
		"""
		mask = bool_proposals.unsqueeze(-1).eq(0)	# [b, proposals, 1]
		# boxes_feat = F.normalize(boxes_feat, p=norm, dim=-1)
		# boxes_feat = torch.cat([boxes_feat, boxes, k_emb], dim=-1)		# here there is also the area
		boxes_feat = torch.cat([boxes_feat, boxes], dim=-1)		# here there is also the area
		boxes_feat = self.img_mlp(boxes_feat)
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
		# head_emb = self.wv(query) # TODO: risolvi

		# mask
		q_emb = q_emb.masked_fill(mask_words, 0)
		k_emb = k_emb.masked_fill(mask_proposals, 0)

		return q_emb, k_emb
	
	def _encode_freezed(self, query, label, attrs, head, bool_words, bool_proposals):
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
		# params
		mask_words = bool_words.unsqueeze(-1).eq(0)
		mask_proposals = bool_proposals.unsqueeze(-1).eq(0)

		bool_head = torch.greater(head, 0).type(torch.long)
		mask_head = bool_head.unsqueeze(-1).eq(0)

		q_emb = self.wv_freezed(query)
		k_emb = self.wv_freezed(label)
		# attr_emb = self.wv_freezed(attrs)
		head_emb = self.wv_freezed(head)

		# mask
		q_emb = q_emb.masked_fill(mask_words, 0)
		k_emb = k_emb.masked_fill(mask_proposals, 0)
		head_emb = head_emb.masked_fill(mask_head, 0)

		return q_emb, k_emb, head_emb

	def _get_query_similarity(self, q_emb, bool_words, bool_queries):
		"""
		:param q_emb: [b, query, words, emb]
		:param bool_words: [b, query, words]
		:param bool_queries: [b, query]
		:return: [b, query, query]
		"""
		batch_size = q_emb.shape[0]

		num_words = torch.sum(bool_words, dim=-1)  # [b, query]
		num_queries = torch.sum(bool_queries, dim=-1)  # [b]

		# averaged query representation over words
		query_repr = torch.sum(q_emb, dim=-2)                                 # [b, query, emb]
		query_repr = query_repr / num_words.unsqueeze(-1)                             # [b, query, emb]
		query_repr = query_repr.masked_fill((num_words == 0).unsqueeze(-1), value=0)  # [b, query, emb]

		# averaged query representation over phrases
		query_repr = torch.sum(query_repr, dim=-2)                                      # [b, emb]
		query_repr = query_repr / num_queries.unsqueeze(-1)                             # [b, emb]
		query_repr = query_repr.masked_fill((num_queries == 0).unsqueeze(-1), value=0)  # [b, emb]

		query_repr_a = query_repr.unsqueeze(1).repeat(1, batch_size, 1)  # [b, b, emb]
		query_repr_b = query_repr.unsqueeze(0).repeat(batch_size, 1, 1)  # [b, b, emb]
		
		query_similarity = self.similarity_function(query_repr_a, query_repr_b)    # [b, b]
					
		return query_similarity


class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, hid_sizes: List[int],
                 activation_function: Callable = F.relu,
                 init_function: Callable = nn.init.xavier_normal_,
                 dropout=0.):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.activation_function = activation_function
        self.init_function = init_function
        self.layers = self.make_network_params()
        self.dropout = nn.Dropout(p=dropout)

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        # definition
        layers = nn.ModuleList([nn.Linear(s[0], s[1]) for (i, s) in enumerate(weight_sizes)])
        # init
        for layer in layers:
            self.init_function(layer.weight)
            nn.init.zeros_(layer.bias)
        return layers

    def forward(self, inputs):
        acts = inputs
        for layer in self.layers:
            hid = layer(acts)
            if self.training:
              hid = self.dropout(hid)
            acts = self.activation_function(hid)
        last_hidden = hid
        return last_hidden
