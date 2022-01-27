train_process_pid=74468

while ps -p $train_process_pid > /dev/null
do
    sleep 60
done

echo "Starting evaluation..."

./run_eval_1.sh&
sleep 1
./run_eval_2.sh&
sleep 1
./run_eval_3.sh
