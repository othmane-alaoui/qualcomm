search_dir=`ls -R -p -lc -I resultats ./Desktop/Raspberry-pi/`
echo $search_dir > ./Desktop/Raspberry-pi/resultats/last-change2.txt
difference=`diff -q "./Desktop/Raspberry-pi/resultats/last-change.txt" "./Desktop/Raspberry-pi/resultats/last-change2.txt"`
rm ./Desktop/Raspberry-pi/resultats/last-change.txt
mv ./Desktop/Raspberry-pi/resultats/last-change2.txt ./Desktop/Raspberry-pi/resultats/last-change.txt

if [ "$difference" = "Les fichiers ./Desktop/Raspberry-pi/resultats/last-change.txt et ./Desktop/Raspberry-pi/resultats/last-change2.txt sont différents" ]
then
    echo "Files have been modified, scripts will start soon"
    sudo bash ./Desktop/Raspberry-pi/scripts/container-setup.sh

else
    echo "--Loading--"
fi