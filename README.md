To run the experiment, put `Data` and `Software` under the same directory.

1. mv Data/* Software/data/dataset
2. cd Software/data
3. CUDA_VISIBLE_DEVICES=0,1 python generate.py
4. cd ..
5. bash script/release.sh $data_name 0

$data_name can be agnews, searchsnippets, stackoverflow, biomedical, googlenews-ts, googlenews-t, googlenews-s, tweet.
