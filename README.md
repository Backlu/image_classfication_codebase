# image_classfication_codebase
a common image classification codebase


### Inference/Training SOP

1. Copy Image to folder 
    1. data folder: /mnt/hdd1/Data/Competition/orchid
2. Detect flower & cut image (程式: flower_detect.ipynb)
    1. 定義data_floder
    2. 執行程式
    3. 處理後的cutted image會存放在 {data_folder}_aug
3. Inference (程式: inference.ipynb)
    1. read image, save in dataframe
    2. 定義model_ver
    3. 執行程式
    4. 分類結果會存放在csv檔: {data_folder}/pseudo_label.csv
    5. generate submit.csv
4. Training
    1. read image, save in dataframe (inclue raw image & cutted images)
    2. 定義tiral_ver, learning rate, pretrain_dir, epoch, candidate_models
    3. 執行程式
    4. 訓練好的model會存放在資料夾: {model_dir}_{tiral_ver}