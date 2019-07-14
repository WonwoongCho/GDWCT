FILE=$1

if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./datasets/celeba.zip
    mkdir -p ./datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE


elif [ $FILE == 'pretrained' ]; then

    # GDWCT trained on CelebA (Bangs<=>No_Bangs, Smiling<=>Non-Smiling), 216x216 resolution
    export fileid=1C3Ru_CnMCYh1W1FxwaztJT1CIiv--68i
    export filename=./pretrained_models/GDWCT_pretrained.zip
    
    mkdir -p ./pretrained_models/
    wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

    wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

    unzip $filename -d ./pretrained_models/
    rm $filename

else
    echo "Available arguments are 'celeba' or 'pretrained'."
    exit 1
fi
