# OpenAI Grok Curve Experiments

# 2.1 Data Generation
`scripts/make_data.py`

# 2.2

`python scripts/train.py`

`
--n_layers 2
--n_heads 4
--d_model
128
--random_seed
0
--datadir
"../data97"
--math_operator
"/"
--logdir
../logs/p97_l2_s0_div
--max_steps
100000
--max_epochs
10000
`