# Controlling params 
setting=$1 #dst, c2t, or e2e
stage=$2 #to select training, test, and eval stage
small_data=$3 # for debugging purpose, only training for 3 epochs, use small sized data only  
version=$4 # 2.0 or 2.1 

# Data params
prefix="updated_"
max_dial_his_len=1
detach_dial_his=1  
add_prev_dial_state=1 
num_workers=4

# Model params
model=tf
sys_act=1 
nb_blocks_res_dec=3
nb_blocks_slot_dst=3
nb_blocks_domain_dst=3
nb_blocks_domain_slot_dst=0
d_model=256
d_ff=$(( d_model*4 ))
att_h=8
dropout=0.3

# Training params 
out_directory=exps
batch_size=32
warmup_steps=10000
if [ $small_data -eq 1 ]; then
  epochs=3
  out_directory=${out_directory}_small
else
  epochs=30
fi

# Pretrained model
pretrained_dst=${out_directory}/multiwoz${version}_dst/${model}_best.pth.tar
# output directory 
out_dir=${out_directory}/multiwoz${version}_${setting}

if [ $stage -le 1 ]; then
python train.py  \
          --out-dir $out_dir \
          --data-version $version \
          --small-data $small_data \
          --num-workers $num_workers \
          --prefix $prefix \
          --model $model \
          --warmup-steps $warmup_steps \
          --dropout $dropout \
          --num-epochs $epochs \
          --batch-size $batch_size \
          --detach-dial-his $detach_dial_his \
          --add-prev-dial-state $add_prev_dial_state \
          --max-dial-his-len $max_dial_his_len \
          --setting $setting \
          --nb-blocks-slot-dst $nb_blocks_slot_dst \
          --nb-blocks-domain-dst $nb_blocks_domain_dst \
          --nb-blocks-domain-slot-dst $nb_blocks_domain_slot_dst \
          --nb-blocks-res-dec $nb_blocks_res_dec \
          --d-model $d_model \
          --d-ff $d_ff \
          --att-h $att_h \
          --pretrained-dst $pretrained_dst \
          --sys-act $sys_act
fi

#testing params
tep='best'
dst_max_len=10
res_max_len=20
res_min_len=1
beam=5
penalty=1.0
nbest=5
gt_db_pointer=0
gt_previous_bs=0
if [ "$setting" = "c2t" ]; then
    gt_db_pointer=1
    gt_previous_bs=1
fi
output=output_gtpointer${gt_db_pointer}_gtprevbs${gt_previous_bs}_ep${tep}.json

if [ $stage -le 2 ]; then
  python generate.py \
          --out-dir $out_dir \
          --model $model \
          --dst-max-len $dst_max_len \
          --res-max-len $res_max_len \
          --res-min-len $res_min_len \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $output \
          --gt-db-pointer $gt_db_pointer \
          --gt-previous-bs $gt_previous_bs \
          --small-data $small_data \
          --tep $tep \
          --verbose 0
fi

if [ $stage -le 3 ]; then
    if [ "$setting" != "c2t" ]; then
        python dst_evaluate.py \
          --out-dir $out_dir \
          --model $model \
          --dst-max-len $dst_max_len \
          --res-max-len $res_max_len \
          --res-min-len $res_min_len \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $output \
          --gt-db-pointer $gt_db_pointer \
          --gt-previous-bs $gt_previous_bs \
          --small-data $small_data \
          --tep $tep
    fi 

    if [ "$setting" != "dst" ]; then
        python c2t_evaluate.py \
          --out-dir $out_dir \
          --model $model \
          --dst-max-len $dst_max_len \
          --res-max-len $res_max_len \
          --res-min-len $res_min_len \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $output \
          --gt-db-pointer $gt_db_pointer \
          --gt-previous-bs $gt_previous_bs \
          --small-data $small_data \
          --tep $tep
    fi
fi
