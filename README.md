# IntermittentML
## HOW TO LOAD THE DATASET

Download the files from this page --> https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Put the unzipped csvs into the Data folder.

run the following to install the required dependencies
```
pip3 install -r src/requirements.txt
```

run the following to load the data into the ML format
```
python3 pre-processing/intermittent_products.py
```

run the following to run the Torch version
```
python3 src/app_no_tensorflow.py
```


To run the Torch version
```
python3 src/app.py
```

Here is how you can change the number of rows you use, the models you run, etc. Edit the `main()` function parameters in the respective script files to adjust settings like `num_rows`, `epochs`, `batch_size`, `hidden_1`, and `hidden_2`. 

To plot the model logs run
```
python3 src/plot_model_logs.py
```
and check the outputs folder.

