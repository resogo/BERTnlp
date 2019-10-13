BERT NLP Files

Meaning of Python Files:
bert_ner_n: attempting bert ner with normal distribution of output layer weights
bert_ner_x: bert ner with xavier initialization for output weights
bert_ner: output weights not specified
create_pretraining_data: used to create pretraining data before running run_pretraining (google)
run_pretraining: used to run pretraining with the created data (google)
summary.py: reads the data from the eval_results.txt file and creates a csv summary and graph
summary_fine.py: creates a csv and graph summary of the pretraining checkpoints as seen in checkpoint_0_graph.png


CSV Files:
Eval: checkpoints evaluated from the fine_out directory
Eval_X: checkpoints evaluated from the fine_out_x directory

Jupyter Notebooks:
Learning: learning how to use matplot lib and code to write sh files to more efficiently run bert
Pytorch-bert-ner: attempted implementation of bert ner with pytorch
Pretraining with MIMIC: preparing MIMIC train.txt and test.txt; in the end the split wasn't necessary and both files were used for pretraining after being split into shards
Word Embeddings: Using BERT to create Word Embeddings
BERT Data Configuration: Formating i2b2 Data for NER with PHI
BERT NER Data Format: Formating i2b2 Data using sentence labels
bert_ner_jupyter: bert ner in jupyter notebook (not up to date)
From Scratch Pre-training: pretraining from scratch
predicting_movie_reviews_with_bert_on_tf_hub: example by google

Current Errors:
   -  getting bert_ner_n and bert_ner_x to work properly and increasing precision and recall
   -  getting an error when trying to multiply output_weights with output_layer
      -  "logits = tf.matmul(output_layer, output_weights, transpose_b=True)" line 430
      -  incompatible shapes
   - has previously worked and I haven't been able to undo what broke it
      - started when I installed a different type of tensorflow and tried to switch from xavier to normal initialization

Created by Rebecca Golm on August 26th, 2019 (last edited October 13th, 2019)