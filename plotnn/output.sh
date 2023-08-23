if [ ! -f "./temp.txt" ]; then touch temp.txt; fi
sudo docker ps > ./temp.txt
container_id=$(sed -n "2,1p" temp.txt | awk '{print $1}')
sudo docker cp $container_id:/home/PlotNeuralNet/model_vis.pdf ./