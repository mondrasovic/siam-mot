screen -dmS eval_1
screen -S eval_1 -X screen ./run_eval_1.sh

screen -dmS eval_2
screen -S eval_2 -X screen ./run_eval_2.sh

screen -dmS eval_3
screen -S eval_3 -X screen ./run_eval_3.sh
