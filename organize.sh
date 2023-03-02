
rm *txt
rm *png
END=25

for n in $(seq 1 $END); do 

	echo "working on ds${n}"
	mkdir "ds${n}"
	cd "ds${n}"
	rm "../dat_ds${n}.pkl_fit_asym"
	rm "../dat_ds${n}.pkl_fit_asym_bin2"
	rm "../dat_ds${n}.pkl_fit_asym_bin4"
	rm "../dat_ds${n}.pkl_fit_asym_bin6"
	rm "../dat_ds${n}.pkl_fit_asym_bin2"
	rm "../dat_ds${n}.pkl_refit_asym"
	rm "../dat_ds${n}.pkl_fit"
	mv "../dat_ds${n}.pkl_fit_bin2_asym" "fitbin2.pkl"
	mv "../dat_ds${n}.pkl" "diskset.pkl"
	mv "../dat_ds${n}.pkl_asym_refit" "fit.pkl"
	mv "../dat_ds${n}.pkl_unwrap" "unwrap.pkl"
	mv "../datredo1_ds${n}.pkl_unwrap" "unwrap-v2.pkl"
	mv "../datredo2_ds${n}.pkl_unwrap" "unwrap-v3.pkl"
	mv "../datredo3_ds${n}.pkl_unwrap" "unwrap-v4.pkl"
	mv "../datredo4_ds${n}.pkl_unwrap" "unwrap-v5.pkl"
	mv "../adjust_ds${n}.pkl_unwrap" "unwrap-adj.pkl"
	cd ..

done




