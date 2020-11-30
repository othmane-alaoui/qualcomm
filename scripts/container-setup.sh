mkdir ./Desktop/Raspberry-pi/test-onnx
mkdir ./Desktop/Raspberry-pi/test-h5
mkdir ./Desktop/Raspberry-pi/test-onnx/resultats
mkdir ./Desktop/Raspberry-pi/test-h5/resultats
cp -r ./Desktop/Raspberry-pi/images ./Desktop/Raspberry-pi/test-onnx
cp -r ./Desktop/Raspberry-pi/images ./Desktop/Raspberry-pi/test-h5
cp ./Desktop/Raspberry-pi/scripts/compare-models-onnx.py ./Desktop/Raspberry-pi/test-onnx
cp ./Desktop/Raspberry-pi/scripts/compare-models-h5.py ./Desktop/Raspberry-pi/test-h5
mkdir ./Desktop/Raspberry-pi/test-onnx/models
mkdir ./Desktop/Raspberry-pi/test-h5/models
search_dir=`ls ./Desktop/Raspberry-pi/models/*`
for entry in $search_dir
do
  extension="${entry##*.}"
  filename="${entry##*/}"
  if [ $extension = "onnx" ]
  then
  	cp "$entry" ./Desktop/Raspberry-pi/test-onnx/models
  else
      if [ $extension = "h5" ]
      then
      	   cp "$entry" ./Desktop/Raspberry-pi/test-h5/models
      fi
  fi
done

container_onnx="container_onnx"
container_h5="container_onnx"

sudo docker start $container_onnx
sudo docker exec $container_onnx rm -rf ./test-model
sudo docker cp ./Desktop/Raspberry-pi/test-onnx $container_onnx:/scripts/test-model
sudo docker stop $container_onnx

sudo docker start $container_h5
sudo docker exec $container_h5 rm -rf ./test-model
sudo docker cp ./Desktop/Raspberry-pi/test-h5 $container_h5:/test-model
sudo docker stop $container_h5

sudo rm -rf ./Desktop/Raspberry-pi/test-onnx
sudo rm -rf ./Desktop/Raspberry-pi/test-h5
