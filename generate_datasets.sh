NS=(10 20 50)
LS=(100 300 500)
MD=("normalized" "first_order" "second_order")
SAVE_DIR="data4"

for ns in ${NS[@]}; do
    for ls in ${LS[@]}; do
        for md in ${MD[@]}; do
            npz="ns${ns}_ls${ls}_${md}.npz"
            echo ${npz}
            python generate_dataset.py --split random_balanced --save-path "${SAVE_DIR}/${npz}" --n-samples ${ns} --len-sample ${ls} --preprocess ${md}
        done
    done
done
echo "Done. "

