NS=(10 20 50)
LS=(100 300 500)
# MD=("normalized" "first_order" "second_order")
MD=("normalized")
SAVE_DIR="data"

for ns in ${NS[@]}; do
    for ls in ${LS[@]}; do
        for md in ${MD[@]}; do
            npz="ns${ns}_ls${ls}_${md}.npz"
            echo ${npz}
            python generate_dataset.py --anno-file rawdata/annotations.csv --split random_balanced --save-path "${SAVE_DIR}/${npz}" --n-samples-train ${ns} --len-sample ${ls} --preprocess ${md} --switch-prob-train -1
        done
    done
done

for ns in ${NS[@]}; do
    for ls in ${LS[@]}; do
        for md in ${MD[@]}; do
            npz="ns${ns}_ls${ls}_${md}_switch.npz"
            echo ${npz}
            python generate_dataset.py --anno-file rawdata/annotations.csv --split random_balanced --save-path "${SAVE_DIR}/${npz}" --n-samples-train ${ns} --len-sample ${ls} --preprocess ${md}
        done
    done
done
echo "Done. "

