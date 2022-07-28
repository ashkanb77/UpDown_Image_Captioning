# UpDown Image Captioning

implementation of Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering paper

# Training
<code>
  ! bash config_dataset.sh
</code>
<br>
<code>
  ! python train.py --n_epochs 3 --use_only_train2014 1 --feature_extractor faster_rcnn
</code>
