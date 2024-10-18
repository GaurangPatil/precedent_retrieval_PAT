#!/bin/bash
echo "Similarity using P:"
for i in {1..10}
do
    echo "P with  k  = $i"
    trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/P_$i"
done
echo "Similarity using CAT alone:"

for i in {1..10}
do
    echo "CAT with  k  = $i"
    trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/CAT_$i"
done

echo "Doc level (J):"
trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/doc_level"

echo "Doc level (CAT alone):"
trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/doc_level_cat"

echo "Doc level (J + CAT) as a single entity:"
trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/doc_level_cat_appended"

for i in {1..9}
do
    echo "P and CAT with alpha  = 0.$i"
    trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/P_and_CAT_alpha_$i"
done

for i in {1..10}
do
    echo "P U CAT with k = $i"
    trec_eval_latest/trec_eval-9.0.7/trec_eval -m bpref -m P.10 -m map -m recip_rank AILA/data/raw/aila-qrel.txt "AILA/data/results/P_U_CAT_$i"
done