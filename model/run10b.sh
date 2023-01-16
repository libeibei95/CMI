CUDA_VISIBLE_DEVICES='0' python run.py --dataset wechat --sampler R --n_interest 16 --interest_type Plus --log_dir log_wechat_abl --patience  10  --w_cl 0.01 --cl_type prob --temp 0.1  >  log.log 2>&1
CUDA_VISIBLE_DEVICES='0' python run.py --dataset wechat --sampler R --n_interest 8 --interest_type Plus --log_dir log_wechat_abl --patience  10  --w_cl 0.01 --cl_type prob --temp 0.1  >  log.log 2>&1
CUDA_VISIBLE_DEVICES='0' python run.py --dataset wechat --sampler R --n_interest 4 --interest_type Plus --log_dir log_wechat_abl --patience  10  --w_cl 0.01 --cl_type prob --temp 0.1  >  log.log 2>&1
CUDA_VISIBLE_DEVICES='0' python run.py --dataset wechat --sampler R --n_interest 2 --interest_type Plus --log_dir log_wechat_abl --patience  10  --w_cl 0.01 --cl_type prob --temp 0.1  >  log.log 2>&
