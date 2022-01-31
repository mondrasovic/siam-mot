train_process_pid_1=108676
train_process_pid_2=108677

while ps -p $train_process_pid_1 > /dev/null
do
    sleep 20
done

while ps -p $train_process_pid_2 > /dev/null
do
    sleep 20
done

echo "Starting evaluation..."

./run_eval_1.sh&
sleep 1
./run_eval_2.sh&
sleep 1
./run_eval_3.sh
