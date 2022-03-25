
test:
	pytest

run:
	nohup python train.py &
	tail -f nohup.out
cnn_ae:
	python CNN_AE.py

dash:
	tensorboard --logdir experiment_logs

clean:
	rm -r experiment_logs/*