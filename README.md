# Weakly-Supervised Visual-Textual Grounding with Semantic Prior Refinement

This folder contains the code used to produce the results reported in the paper: [Weakly-Supervised Visual-Textual Grounding with Semantic Prior Refinement]() 

Some of our code is based on [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework) . Thanks!

# Dependencies
This project uses the `conda` environment.
In the `root` folder you can find the `.yml` file for the configuration of the `conda` environment and also the `.txt` files for the `pip` environment. 

# Structure
The project is structured as follows: 
* `data`: contains datasets and pre-processed files;
* `model`: contains code about the model;
* `output`: contains checkpoints and the results.
* `utils`: contains some code adopted in model.

# Usage
NOTE: in order to execute correctly the code, the users need to set in the code their absolute path to this folder: `SPRM-project`.

### Proposal Extraction
For Flickr30k Entities we ha adopted the pre-processed dataset publicly available in this repository: [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework).
For this reason, we do not have extracted ourself the proposals.

Regarding ReferIt, which is not included in the manuscript's assessment of [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework), we have adopted the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) object detector.
In particular, we have adopted this Pytorch implementation: https://github.com/MILVLG/bottom-up-attention.pytorch
The features are extracted with confidence of 0.2.
The extracted features take up too much space and we are unable to upload them online. For this reason we upload the pre-processed files directly (see sections below).

### Data Download
First you need to download the necessary datasets. In particular, it is needed Flickr30k Entities dataset and ReferIt dataset, respectively.
The final structure should be:
```
Loss_VT_Grounding
|-- data
    |-- flickr30k
        |-- flickr30k_images    # all Flickr30k images
        |-- flickr30k_entities
            |-- Flickr30kEntities
                |-- Annotations
                |-- Sentences
                |-- ...
        |-- test.txt
        |-- test_detection_dict.json
        |-- test_features_compress.hdf5
        |-- test_images_size.json
        |-- test_imgid2idx.pkl.json
        |-- train.txt
        |-- train_detection_dict.json
        |-- train_features_compress.hdf5
        |-- train_images_size.json
        |-- train_imgid2idx.pkl.json
        |-- val.txt
        |-- val_detection_dict.json
        |-- val_features_compress.hdf5
        |-- val_images_size.json
        |-- val_imgid2idx.pkl.json
    |-- referit
        |-- refer
            |-- data
            |-- evaluation
            |-- external
            |-- ...
        |-- test.txt
        |-- test_detection_dict.json
        |-- test_features_compress.hdf5
        |-- test_images_size.json
        |-- test_imgid2idx.pkl.json
        |-- test_referit_resnet101_faster_rcnn_genome.tsv
        |-- train.txt
        |-- train_detection_dict.json
        |-- train_features_compress.hdf5
        |-- train_images_size.json
        |-- train_imgid2idx.pkl.json
        |-- train_referit_resnet101_faster_rcnn_genome.tsv
        |-- val.txt
        |-- val_detection_dict.json
        |-- val_features_compress.hdf5
        |-- val_images_size.json
        |-- val_imgid2idx.pkl.json
        |-- val_referit_resnet101_faster_rcnn_genome.tsv
    |-- glove
        |-- glove.6B.300d.txt
```

Where:
1) The glove embeddings can be download here: http://nlp.stanford.edu/data/glove.6B.zip
2) `refer` is the following repository: [https://github.com/lichengunc/refer](https://github.com/lichengunc/refer). 
The user just need to download the images from http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip and unzip them in folder `./data/referit/refer/data/images/saiapr_tc-12`.


### Environment
To configure the environment:
```bash
conda env create -f env.yml 
conda activate SPRM
pip install -r env.txt
```

### Data Pre-processing
For Fickr30k Entities dataset, we have adopted the same pre-processed features of [MAF](https://github.com/qinzzz/Multimodal-Alignment-Framework).
Concerning ReferIt, given the extracted proposals type the following commands to generate the final pre-processed data:
```bash
python make_dataset_tsv_bu-d2.py --extracted_features ./data/referit/extracted_features/
python make_dataset_h5py.py --task referit
python make_dataset_obj_dict_bu-d2.py --extracted_features ./data/referit/extracted_features/
```
The generated files are placed in `./data/referit/`.

### Model Training
In order to train the model use:
```bash
python main.py  --dataset flickr30k   \                       # dataset name [flickr30k, referit]
                --batch 16   \                                # batch size
                --cosine_weight 0.4  \                        # omega hyper-parameter
                --device cuda   \                             # device [cuda, cpu]
                --cosine_similarity_strategy mean    \        # aggregation strategy. In the paper we adopt mean
                --loss_strategy neg1_max   \                  # Our negative contrastive loss
                --train_fract 1   \                           # fraction fo training set (0, ..., 1]
                --do_oov   \                                  # spell checking
                --do_head   \                                 # extract noun phrase's head
                --use_head_for_concept_embedding  \           # use head in the concept branch
                --do_relations  \                             # use boxes relations
                --do_locations   \                            # use noun phrase location
                --use_relations_for_concept_embedding   \     # apply spatial relation
                --relations_strategy baseline   \             # algorithm of labeling the spatial locations
                --use_spatial_features   \                    # use normalized spatial coordinates
```
All the available parameters can be seen typing:
```
python main.py --help
```

### Model Test
In order to test the model:
```bash
python main.py  --dataset flickr30k   \                       # dataset name [flickr30k, referit]
                --batch 16   \                                # batch size
                --cosine_weight 0.4   \                       # omega hyper-parameter
                --device cuda   \                             # device [cuda, cpu]
                --cosine_similarity_strategy mean    \        # aggregation strategy. In the paper we adopt mean
                --loss_strategy neg1_max   \                  # Our negative contrastive loss
                --train_fract 1   \                           # fraction fo training set (0, ..., 1]
                --do_oov   \                                  # spell checking
                --do_head   \                                 # extract noun phrase's head
                --use_head_for_concept_embedding  \           # use head in the concept branch
                --do_relations  \                             # use boxes relations
                --do_locations   \                            # use noun phrase location
                --use_relations_for_concept_embedding   \     # apply spatial relation
                --relations_strategy baseline   \             # algorithm of labeling the spatial locations
                --use_spatial_features   \                    # use normalized spatial coordinates
                --test_set   \                                # load test set instead of validation set
                --dry_run   \                                 # load only test set
                --file ./output/flickr/best_omega1.pt \       # load checkpoint
```
All the available parameters can be seen typing:
```
python main.py --help
```

# Pre-processed datasets, Pre-trained Models and Results
To download the pre-trained weights: [https://drive.google.com/file/d/1a2NW_v_XouHNB7LTIrSVRDni3O5i135j/view?usp=share_link](https://drive.google.com/file/d/1a2NW_v_XouHNB7LTIrSVRDni3O5i135j/view?usp=share_link).

To download the ReferIt pre-processed dataset: [https://drive.google.com/file/d/1nJiN5jSP9tF0MJwQOkpbvI1YPsTySSoD/view?usp=share_link](https://drive.google.com/file/d/1nJiN5jSP9tF0MJwQOkpbvI1YPsTySSoD/view?usp=share_link)
