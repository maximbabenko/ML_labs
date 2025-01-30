echo "----Install Dependencies (begin)-----"
pip install -r requirements.txt
echo "----Install Dependencies (end)-----"

echo "----Create Dataset (begin)-----"
python /home/user/PROJECTS/urfu/MLOps_1/lab2/create_dataset.py 
echo "----Create Dataset (end)-----"

echo "----Data Preprocessing (begin)-----"
python /home/user/PROJECTS/urfu/MLOps_1/lab2/data_preprocessing.py
echo "----Data Preprocessing (end)-----"

echo "----Train the Model (begin)-----"
python /home/user/PROJECTS/urfu/MLOps_1/lab2/model_training.py
echo "----Train the Model (end)-----"

echo "----Use the Model for Prediction (begin)-----"
python /home/user/PROJECTS/urfu/MLOps_1/lab2/model_testing.py
echo "----Use the Model for Prediction (begin)-----"