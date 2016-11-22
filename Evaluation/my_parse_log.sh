#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi
LOG=`basename $1`
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt
sed -i '/Waiting for data/d' aux.txt
sed -i '/prefetch queue empty/d' aux.txt
sed -i '/Iteration .* loss/d' aux.txt
sed -i '/Iteration .* lr/d' aux.txt
sed -i '/Train net/d' aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep 'Test net output #0:' aux.txt | awk '{print $11}' > aux1.txt
grep 'Test net output #1:' aux.txt | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux3.txt
grep 'Testing net' $1 >> aux3.txt
python extract_seconds.py aux3.txt aux4.txt
grep 'Test net output #4: loss_5_o_Clock_Shadow = ' $1 | awk '{print $11}' > aux5.txt
grep 'Test net output #5: loss_Arched_Eyebrows = ' $1 | awk '{print $11}' > aux6.txt
grep 'Test net output #6: loss_Attractive = ' $1 | awk '{print $11}' > aux7.txt
grep 'Test net output #7: loss_Bags_Under_Eyes = ' $1 | awk '{print $11}' > aux8.txt
grep 'Test net output #8: loss_Bald = ' $1 | awk '{print $11}' > aux9.txt
grep 'Test net output #9: loss_Bangs = ' $1 | awk '{print $11}' > aux10.txt
grep 'Test net output #10: loss_Big_Lips = ' $1 | awk '{print $11}' > aux11.txt
grep 'Test net output #11: loss_Big_Nose = ' $1 | awk '{print $11}' > aux12.txt
grep 'Test net output #12: loss_Black_Hair = ' $1 | awk '{print $11}' > aux13.txt
grep 'Test net output #13: loss_Blond_Hair = ' $1 | awk '{print $11}' > aux14.txt
grep 'Test net output #14: loss_Blurry = ' $1 | awk '{print $11}' > aux15.txt
grep 'Test net output #15: loss_Brown_Hair = ' $1 | awk '{print $11}' > aux16.txt
grep 'Test net output #16: loss_Bushy_Eyebrows = ' $1 | awk '{print $11}' > aux17.txt
grep 'Test net output #17: loss_Chubby = ' $1 | awk '{print $11}' > aux18.txt
grep 'Test net output #18: loss_Double_Chin = ' $1 | awk '{print $11}' > aux19.txt
grep 'Test net output #19: loss_Eyeglasses = ' $1 | awk '{print $11}' > aux20.txt
grep 'Test net output #20: loss_Goatee = ' $1 | awk '{print $11}' > aux21.txt
grep 'Test net output #21: loss_Gray_Hair = ' $1 | awk '{print $11}' > aux22.txt
grep 'Test net output #22: loss_Heavy_Makeup = ' $1 | awk '{print $11}' > aux23.txt
grep 'Test net output #23: loss_High_Cheekbones = ' $1 | awk '{print $11}' > aux24.txt
grep 'Test net output #24: loss_Male = ' $1 | awk '{print $11}' > aux25.txt
grep 'Test net output #25: loss_Mouth_Slightly_Open = ' $1 | awk '{print $11}' > aux26.txt
grep 'Test net output #26: loss_Mustache = ' $1 | awk '{print $11}' > aux27.txt
grep 'Test net output #27: loss_Narrow_Eyes = ' $1 | awk '{print $11}' > aux28.txt
grep 'Test net output #28: loss_No_Beard = ' $1 | awk '{print $11}' > aux29.txt
grep 'Test net output #29: loss_Oval_Face = ' $1 | awk '{print $11}' > aux30.txt
grep 'Test net output #30: loss_Pale_Skin = ' $1 | awk '{print $11}' > aux31.txt
grep 'Test net output #31: loss_Pointy_Nose = ' $1 | awk '{print $11}' > aux32.txt
grep 'Test net output #32: loss_Receding_Hairline = ' $1 | awk '{print $11}' > aux33.txt
grep 'Test net output #33: loss_Rosy_Cheeks = ' $1 | awk '{print $11}' > aux34.txt
grep 'Test net output #34: loss_Sideburns = ' $1 | awk '{print $11}' > aux35.txt
grep 'Test net output #35: loss_Smiling = ' $1 | awk '{print $11}' > aux36.txt
grep 'Test net output #36: loss_Straight_Hair = ' $1 | awk '{print $11}' > aux37.txt
grep 'Test net output #37: loss_Wavy_Hair = ' $1 | awk '{print $11}' > aux38.txt
grep 'Test net output #38: loss_Wearing_Earrings = ' $1 | awk '{print $11}' > aux39.txt
grep 'Test net output #39: loss_Wearing_Hat = ' $1 | awk '{print $11}' > aux40.txt
grep 'Test net output #40: loss_Wearing_Lipstick = ' $1 | awk '{print $11}' > aux41.txt
grep 'Test net output #41: loss_Wearing_Necklace = ' $1 | awk '{print $11}' > aux42.txt
grep 'Test net output #42: loss_Wearing_Necktie = ' $1 | awk '{print $11}' > aux43.txt
grep 'Test net output #43: loss_Young = ' $1 | awk '{print $11}' > aux44.txt
grep 'Test net output #44: loss_mid = ' $1 | awk '{print $11}' > aux45.txt
grep 'Test net output #45: loss_top = ' $1 | awk '{print $11}' > aux46.txt

# Generating
paste aux0.txt aux4.txt aux1.txt aux2.txt aux5.txt aux6.txt aux7.txt aux8.txt aux9.txt aux10.txt \
aux11.txt aux12.txt aux13.txt aux14.txt aux15.txt aux16.txt aux17.txt aux18.txt aux19.txt aux20.txt \
aux21.txt aux22.txt aux23.txt aux24.txt aux25.txt aux26.txt aux27.txt aux28.txt aux29.txt aux30.txt \
aux31.txt aux32.txt aux33.txt aux34.txt aux35.txt aux36.txt aux37.txt aux38.txt aux39.txt aux40.txt \
aux41.txt aux42.txt aux43.txt aux44.txt aux45.txt aux46.txt | column -t >> $LOG.test
rm aux.txt aux0.txt aux4.txt aux1.txt aux2.txt aux5.txt aux6.txt aux7.txt aux8.txt aux9.txt aux10.txt \
aux11.txt aux12.txt aux13.txt aux14.txt aux15.txt aux16.txt aux17.txt aux18.txt aux19.txt aux20.txt \
aux21.txt aux22.txt aux23.txt aux24.txt aux25.txt aux26.txt aux27.txt aux28.txt aux29.txt aux30.txt \
aux31.txt aux32.txt aux33.txt aux34.txt aux35.txt aux36.txt aux37.txt aux38.txt aux39.txt aux40.txt \
aux41.txt aux42.txt aux43.txt aux44.txt aux45.txt aux46.txt

# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ', loss = ' $1 | awk '{print $9}' > aux1.txt
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt
grep 'Train net output #0: ' $1 | awk '{print $11}' > aux5.txt
grep 'Train net output #1: ' $1 | awk '{print $11}' > aux6.txt
grep 'Train net output #2: ' $1 | awk '{print $11}' > aux7.txt
grep 'Train net output #3: ' $1 | awk '{print $11}' > aux8.txt
grep 'Train net output #4: ' $1 | awk '{print $11}' > aux9.txt
grep 'Train net output #5: ' $1 | awk '{print $11}' > aux10.txt
grep 'Train net output #6: ' $1 | awk '{print $11}' > aux11.txt
grep 'Train net output #7: loss_Big_Nose = ' $1 | awk '{print $11}' > aux12.txt
grep 'Train net output #8: loss_Black_Hair = ' $1 | awk '{print $11}' > aux13.txt
grep 'Train net output #9: loss_Blond_Hair = ' $1 | awk '{print $11}' > aux14.txt
grep 'Train net output #10: loss_Blurry = ' $1 | awk '{print $11}' > aux15.txt
grep 'Train net output #11: loss_Brown_Hair = ' $1 | awk '{print $11}' > aux16.txt
grep 'Train net output #12: loss_Bushy_Eyebrows = ' $1 | awk '{print $11}' > aux17.txt
grep 'Train net output #13: loss_Chubby = ' $1 | awk '{print $11}' > aux18.txt
grep 'Train net output #14: loss_Double_Chin = ' $1 | awk '{print $11}' > aux19.txt
grep 'Train net output #15: loss_Eyeglasses = ' $1 | awk '{print $11}' > aux20.txt
grep 'Train net output #16: loss_Goatee = ' $1 | awk '{print $11}' > aux21.txt
grep 'Train net output #17: loss_Gray_Hair = ' $1 | awk '{print $11}' > aux22.txt
grep 'Train net output #18: loss_Heavy_Makeup = ' $1 | awk '{print $11}' > aux23.txt
grep 'Train net output #19: loss_High_Cheekbones = ' $1 | awk '{print $11}' > aux24.txt
grep 'Train net output #20: loss_Male = ' $1 | awk '{print $11}' > aux25.txt
grep 'Train net output #21: loss_Mouth_Slightly_Open = ' $1 | awk '{print $11}' > aux26.txt
grep 'Train net output #22: loss_Mustache = ' $1 | awk '{print $11}' > aux27.txt
grep 'Train net output #23: loss_Narrow_Eyes = ' $1 | awk '{print $11}' > aux28.txt
grep 'Train net output #24: loss_No_Beard = ' $1 | awk '{print $11}' > aux29.txt
grep 'Train net output #25: loss_Oval_Face = ' $1 | awk '{print $11}' > aux30.txt
grep 'Train net output #26: loss_Pale_Skin = ' $1 | awk '{print $11}' > aux31.txt
grep 'Train net output #27: loss_Pointy_Nose = ' $1 | awk '{print $11}' > aux32.txt
grep 'Train net output #28: loss_Receding_Hairline = ' $1 | awk '{print $11}' > aux33.txt
grep 'Train net output #29: loss_Rosy_Cheeks = ' $1 | awk '{print $11}' > aux34.txt
grep 'Train net output #30: loss_Sideburns = ' $1 | awk '{print $11}' > aux35.txt
grep 'Train net output #31: loss_Smiling = ' $1 | awk '{print $11}' > aux36.txt
grep 'Train net output #32: loss_Straight_Hair = ' $1 | awk '{print $11}' > aux37.txt
grep 'Train net output #33: loss_Wavy_Hair = ' $1 | awk '{print $11}' > aux38.txt
grep 'Train net output #34: loss_Wearing_Earrings = ' $1 | awk '{print $11}' > aux39.txt
grep 'Train net output #35: loss_Wearing_Hat = ' $1 | awk '{print $11}' > aux40.txt
grep 'Train net output #36: loss_Wearing_Lipstick = ' $1 | awk '{print $11}' > aux41.txt
grep 'Train net output #37: loss_Wearing_Necklace = ' $1 | awk '{print $11}' > aux42.txt
grep 'Train net output #38: loss_Wearing_Necktie = ' $1 | awk '{print $11}' > aux43.txt
grep 'Train net output #39: loss_Young = ' $1 | awk '{print $11}' > aux44.txt
grep 'Train net output #40: loss_mid = ' $1 | awk '{print $11}' > aux45.txt
grep 'Train net output #41: loss_top = ' $1 | awk '{print $11}' > aux46.txt




# Extracting elapsed seconds
python extract_seconds.py aux.txt aux3.txt

# Generating
paste aux0.txt aux3.txt aux1.txt aux2.txt aux5.txt aux6.txt aux7.txt aux8.txt aux9.txt aux10.txt \
aux11.txt aux12.txt aux13.txt aux14.txt aux15.txt aux16.txt aux17.txt aux18.txt aux19.txt aux20.txt \
aux21.txt aux22.txt aux23.txt aux24.txt aux25.txt aux26.txt aux27.txt aux28.txt aux29.txt aux30.txt \
aux31.txt aux32.txt aux33.txt aux34.txt aux35.txt aux36.txt aux37.txt aux38.txt aux39.txt aux40.txt \
aux41.txt aux42.txt aux43.txt aux44.txt aux45.txt aux46.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux3.txt aux1.txt aux2.txt aux5.txt aux6.txt aux7.txt aux8.txt aux9.txt aux10.txt \
aux11.txt aux12.txt aux13.txt aux14.txt aux15.txt aux16.txt aux17.txt aux18.txt aux19.txt aux20.txt \
aux21.txt aux22.txt aux23.txt aux24.txt aux25.txt aux26.txt aux27.txt aux28.txt aux29.txt aux30.txt \
aux31.txt aux32.txt aux33.txt aux34.txt aux35.txt aux36.txt aux37.txt aux38.txt aux39.txt aux40.txt \
aux41.txt aux42.txt aux43.txt aux44.txt aux45.txt aux46.txt 
