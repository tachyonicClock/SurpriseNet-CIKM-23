
test:
	pytest

run:
	nohup python train.py &
	tail -f nohup.out

dash:
	tensorboard --logdir experiment_logs